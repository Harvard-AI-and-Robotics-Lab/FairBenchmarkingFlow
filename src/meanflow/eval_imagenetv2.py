"""Evaluate a MeanFlow checkpoint on ImageNetV2 (matched-frequency).

Generates class-conditioned samples using CFG at inference, decodes latents
via VAE, and computes FID, Inception Score, CLIP Score, and PickScore against real ImageNetV2 matched-frequency images.

Uses a two-phase approach to avoid GPU OOM: first generates all latents
(model params on GPU), frees the model, then loads the VAE to decode.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenetv2.py \
        --data_path=/path/to/ImageNetV2 \
        --checkpoint_path=weights/checkpoint_1200960/ \
        --num_steps=1 \
        --num_gpus=3
"""
import csv
import gc
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Flax classes are deprecated.*")
os.environ.setdefault('CUDA_VISIBLE_DEVICES_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')  # let JAX release memory for PyTorch

# Parse --num_gpus early (before JAX/PyTorch init) to set CUDA_VISIBLE_DEVICES.
# The cluster scheduler sets CUDA_VISIBLE_DEVICES to the allocated GPUs
# (e.g. "3,5,7"). --num_gpus trims that to the first N entries.
def _parse_num_gpus():
    for arg in sys.argv[1:]:
        if arg.startswith('--num_gpus='):
            return int(arg.split('=', 1)[1])
    return 0

_NUM_GPUS = _parse_num_gpus()
if _NUM_GPUS > 0:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        # Cluster: trim scheduler-allocated GPUs to the first N
        allocated = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(allocated[:_NUM_GPUS])
    else:
        # Local: restrict to the first N GPU IDs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(_NUM_GPUS))
from functools import partial
from pathlib import Path

import jax

if os.environ.get("COORDINATOR_ADDRESS") is not None:
    jax.distributed.initialize()

import jax.numpy as jnp
import ml_collections
import numpy as np
import torch
# PyTorch is only used for CPU data loading; keep it off GPU during generation
_original_cuda_is_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False
import torch.utils.data
from absl import app, flags, logging
from flax import jax_utils
from jax import random

from meanflow import MeanFlow
from utils.logging_util import log_for_0, supress_checkpt_info
from utils.vae_util import LatentManager

# Add src/ to import path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataloader import ImageNetV2Dataset
from utils import eval as eval_metrics

supress_checkpt_info()
warnings.filterwarnings("ignore")

# ======================================================================
# CLI Flags
# ======================================================================

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', None, 'Path to ImageNetV2 root (contains imagenetv2-matched-frequency-format-val/)')
flags.DEFINE_string('checkpoint_path', None, 'Path to MeanFlow checkpoint directory')
flags.DEFINE_string('model_cls', 'DiT_B_4', 'DiT model variant')
flags.DEFINE_string('vae_type', 'mse', 'VAE variant (mse or ema)')
flags.DEFINE_integer('device_batch_size', 1, 'Per-device batch size for sampling')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_boolean('no_ema', False, 'Use raw params instead of EMA params')
flags.DEFINE_integer('num_steps', 1, 'Number of sampling steps')
flags.DEFINE_float('omega', 1.0, 'Guidance strength (cfg_scale)')
flags.DEFINE_boolean('test', False, '[unused] Always evaluates on the full v2 set')
flags.DEFINE_integer('num_gpus', 0, 'Number of GPUs to use (0 = use all available)')
flags.DEFINE_boolean('resume', False, 'Resume from checkpoint (skip completed phases)')
flags.DEFINE_integer('checkpoint_every', 50, 'Save latent checkpoint every N generation steps')
flags.DEFINE_string('model_name', 'meanflow', 'Name used for output directory and file naming')
flags.DEFINE_boolean('compute_fid_IS_per_class', False,
                     'Compute per-class FID and IS (expensive — off by default)')


# ======================================================================
# CFG Generation (mirrors eval_mscoco.py pattern)
# ======================================================================

def generate_with_cfg(variable, model, rng, n_sample, config, labels, cfg_scale, num_classes):
    """Generate samples with classifier-free guidance at inference.

    Calls u_fn twice per step (conditioned + unconditioned) and interpolates:
        u_guided = u_uncond + cfg_scale * (u_cond - u_uncond)

    Args:
        variable: Model parameters dict {"params": ...}.
        model: MeanFlow module instance.
        rng: JAX PRNG key.
        n_sample: Number of samples to generate.
        config: Config dict with sampling/dataset settings.
        labels: (n_sample,) integer array of class labels for conditioning.
        cfg_scale: Guidance scale. 1.0 = no guidance (pure conditional).
        num_classes: Number of classes (null label index = num_classes).

    Returns:
        (n_sample, H, W, C) generated latent array in NHWC format.
    """
    num_steps = config.sampling.num_steps
    img_size = config.dataset.image_size
    img_channels = config.dataset.image_channels

    t_steps = model.apply({}, method=model.sampling_schedule())

    x_shape = (n_sample, img_size, img_size, img_channels)
    rng_xt, rng = jax.random.split(rng, 2)
    z_t = jax.random.normal(rng_xt, x_shape, dtype=model.dtype)

    null_labels = jnp.full((n_sample,), num_classes, dtype=jnp.int32)

    def cfg_step_fn(i, inputs):
        x_i, rng = inputs
        rng_step = jax.random.fold_in(rng, i)
        rng_cond, rng_uncond = jax.random.split(rng_step)

        t = t_steps[i]
        r = t_steps[i + 1]
        t_batch = jnp.repeat(t, n_sample)
        r_batch = jnp.repeat(r, n_sample)
        h = t_batch - r_batch

        u_cond = model.apply(
            variable, x_i, t=t_batch, h=h, y=labels, train=False,
            method=model.u_fn,
            rngs=dict(gen=rng_cond),
        )

        u_uncond = model.apply(
            variable, x_i, t=t_batch, h=h, y=null_labels, train=False,
            method=model.u_fn,
            rngs=dict(gen=rng_uncond),
        )

        u_guided = u_uncond + cfg_scale * (u_cond - u_uncond)
        x_next = x_i - jnp.einsum('n,n...->n...', h, u_guided)

        return (x_next, rng)

    outputs = jax.lax.fori_loop(0, num_steps, cfg_step_fn, (z_t, rng))
    return outputs[0]


def sample_step_cfg(variable, sample_idx, labels, model, rng_init, device_batch_size,
                    config, cfg_scale, num_classes):
    """Per-device sampling step with CFG. Used inside jax.pmap.

    labels: (device_batch_size,) int32 — v2 dataset class indices for this device.
    """
    rng_sample = random.fold_in(rng_init, sample_idx)

    images = generate_with_cfg(
        variable, model, rng_sample,
        n_sample=device_batch_size,
        config=config,
        labels=labels,
        cfg_scale=cfg_scale,
        num_classes=num_classes,
    )
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW (for VAE decode)
    return images, labels


# ======================================================================
# Config
# ======================================================================

def get_eval_config(
    checkpoint_path,
    model_cls='DiT_B_4',
    vae_type='mse',
    num_samples=10000,
    device_batch_size=1,
    seed=42,
    num_steps=1,
    omega=1.0,
):
    config = ml_collections.ConfigDict()

    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.name = 'imagenet'
    dataset.image_size = 32       # latent space (256 // 8)
    dataset.raw_image_size = 256
    dataset.image_channels = 4
    dataset.num_classes = 1000
    dataset.vae = vae_type

    config.training = training = ml_collections.ConfigDict()
    training.seed = seed
    training.adam_b2 = 0.95
    training.ema_val = 0.9999

    config.method = method = ml_collections.ConfigDict()
    method.noise_dist = 'logit_normal'
    method.P_mean = -0.4
    method.P_std = 1.0
    method.data_proportion = 0.75
    method.class_dropout_prob = 0.1
    method.guidance_eq = 'cfg'
    method.omega = omega
    method.kappa = 0.5
    method.t_start = 0.0
    method.t_end = 1.0
    method.norm_p = 1.0
    method.norm_eps = 0.01

    config.model = model = ml_collections.ConfigDict()
    model.cls = model_cls

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.seed = 0
    sampling.num_steps = num_steps
    sampling.schedule = 'default'
    sampling.sampling_timesteps = None
    sampling.num_classes = dataset.num_classes

    config.fid = fid = ml_collections.ConfigDict()
    fid.num_samples = num_samples
    fid.device_batch_size = device_batch_size

    config.load_from = checkpoint_path

    return config


# ======================================================================
# Main Evaluation
# ======================================================================

def _save_metrics(metrics, csv_path):
    """Write metrics dict to CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])


def _compute_all_metrics(generated_images, all_labels, real_images, class_names,
                         real_class_labels=None, compute_fid_IS_per_class=False,
                         cache_dir=None):
    """Compute all evaluation metrics and return as (global_dict, per_class_dict).

    When *cache_dir* is provided each metric is saved to a JSON file as
    soon as it finishes.  On a subsequent call with the same *cache_dir*,
    already-cached metrics are loaded instead of recomputed, so a crash
    mid-way through doesn't lose earlier results.
    """
    cache_path = os.path.join(cache_dir, 'metrics_cache.json') if cache_dir else None
    cached = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)

    _per_class_cache_keys = ('clip_score_per_class', 'pick_score_per_class',
                             'fid_per_class', 'is_per_class')

    def _cache(results, per_class):
        if cache_path:
            data = dict(results)
            data['clip_score_per_class'] = per_class.get('clip_score', {})
            data['pick_score_per_class'] = per_class.get('pick_score', {})
            data['fid_per_class'] = per_class.get('fid', {})
            data['is_per_class'] = per_class.get('inception_score', {})
            with open(cache_path, 'w') as f:
                json.dump(data, f)

    results = {k: v for k, v in cached.items() if k not in _per_class_cache_keys}
    per_class = {}

    if 'fid' not in results:
        log_for_0('Computing FID...')
        results['fid'] = float(eval_metrics.compute_fid(real_images, generated_images))
        _cache(results, per_class)
    else:
        log_for_0(f'FID: cached ({results["fid"]:.4f})')

    if 'is_mean' not in results:
        log_for_0('Computing Inception Score...')
        is_mean, is_std = eval_metrics.compute_inception_score(generated_images)
        results['is_mean'] = float(is_mean)
        results['is_std'] = float(is_std)
        _cache(results, per_class)
    else:
        log_for_0(f'IS: cached ({results["is_mean"]:.4f})')

    if compute_fid_IS_per_class and real_class_labels is not None:
        if 'fid_per_class' not in cached or not cached['fid_per_class']:
            log_for_0('Computing per-class FID...')
            fid_pc = eval_metrics.compute_fid_per_class(
                real_images, real_class_labels, generated_images, all_labels
            )
            per_class['fid'] = {str(k): v for k, v in fid_pc.items()}
            _cache(results, per_class)
        else:
            log_for_0('FID per-class: cached')
            per_class['fid'] = {str(k): v for k, v in cached['fid_per_class'].items()}

        if 'is_per_class' not in cached or not cached['is_per_class']:
            log_for_0('Computing per-class IS...')
            is_pc = eval_metrics.compute_inception_score_per_class(
                generated_images, all_labels
            )
            per_class['inception_score'] = {str(k): v for k, v in is_pc.items()}
            _cache(results, per_class)
        else:
            log_for_0('IS per-class: cached')
            per_class['inception_score'] = {str(k): v for k, v in cached['is_per_class'].items()}

    if 'clip_score' not in results:
        log_for_0('Computing CLIP Score...')
        clip_mean, clip_per_class = eval_metrics.compute_clip_score(
            generated_images, all_labels, class_names, return_per_class=True
        )
        results['clip_score'] = float(clip_mean)
        per_class['clip_score'] = {str(k): v for k, v in clip_per_class.items()}
        _cache(results, per_class)
    else:
        log_for_0(f'CLIP Score: cached ({results["clip_score"]:.4f})')
        per_class['clip_score'] = {str(k): v for k, v in cached.get('clip_score_per_class', {}).items()}

    prompts = [f"a photo of a {class_names[l]}" for l in all_labels]

    if 'pick_score' not in results:
        log_for_0('Computing PickScore...')
        pick_mean, pick_per_class = eval_metrics.compute_pick_score(
            generated_images, prompts, class_labels=all_labels, return_per_class=True
        )
        results['pick_score'] = float(pick_mean)
        per_class['pick_score'] = {str(k): v for k, v in pick_per_class.items()}
        _cache(results, per_class)
    else:
        log_for_0(f'PickScore: cached ({results["pick_score"]:.4f})')
        per_class['pick_score'] = {str(k): v for k, v in cached.get('pick_score_per_class', {}).items()}

    return results, per_class


def evaluate(config, data_path, cfg_scale, use_ema, resume=False,
             checkpoint_every=50, model_name='meanflow'):
    """Main evaluation: generate latents -> decode -> load reference -> metrics -> save."""
    from tqdm import tqdm

    image_size = config.dataset.image_size
    num_classes = config.dataset.num_classes
    num_steps = config.sampling.num_steps

    # ---- Output paths ----
    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = os.path.join(repo_root, 'outputs', 'meanflow', f'steps_{num_steps}')
    subfolder_name = f'{model_name}_steps_{num_steps}_imagenetv2'
    plot_subdir = os.path.join(output_dir, subfolder_name)
    ckpt_dir = os.path.join(plot_subdir, '_checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    latents_path = os.path.join(ckpt_dir, 'latents.npz')
    decoded_path = os.path.join(ckpt_dir, 'decoded.npz')
    partial_latents_path = os.path.join(ckpt_dir, 'latents_partial.npz')

    # ---- Compute step counts for the unified progress bar ----
    samples_per_gen_step = config.fid.device_batch_size * jax.device_count()
    num_gen_steps = int(np.ceil(config.fid.num_samples / samples_per_gen_step))
    decode_batch = config.fid.device_batch_size * jax.local_device_count()
    est_num_decode_steps = int(np.ceil(config.fid.num_samples / decode_batch))

    skip_gen = resume and os.path.exists(latents_path)
    skip_decode = resume and os.path.exists(decoded_path)

    total_steps = (0 if skip_gen else num_gen_steps) + (0 if skip_decode else est_num_decode_steps)
    pbar = tqdm(total=total_steps, desc='Generate', unit='step', dynamic_ncols=True)

    # ==================================================================
    # Phase 1: Generate all latents with CFG (model on GPU)
    # ==================================================================
    if skip_gen:
        log_for_0(f'Phase 1: SKIPPED (resuming from {latents_path})')
        data = np.load(latents_path)
        all_latents = data['latents']
        all_labels = data['labels']
        log_for_0(f'  Loaded {all_latents.shape[0]} latent samples.')
    else:
        # ---- Model ----
        model_config = config.model.to_dict()
        model_str = model_config.pop('cls')

        model = MeanFlow(
            model_str=model_str,
            model_config=model_config,
            **config.sampling,
            **config.method,
        )

        # ---- Load checkpoint ----
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt_path = str(Path(config.load_from).resolve())
        log_for_0(f'Loading checkpoint from {ckpt_path}...')
        raw_ckpt = checkpointer.restore(ckpt_path)

        if use_ema and 'ema_params' in raw_ckpt:
            params = raw_ckpt['ema_params']
            log_for_0('Using EMA params.')
        else:
            params = raw_ckpt['params']
            log_for_0('Using raw params.')
        del raw_ckpt

        params = jax.tree.map(lambda x: np.asarray(x), params)
        state = jax_utils.replicate({"params": params})
        del params

        # ---- Load v2 labels — used to drive class-conditional generation ----
        _meta = ImageNetV2Dataset(data_path=data_path,
                                  image_size=config.dataset.raw_image_size, return_uint8=True)
        val_labels_all = np.array([s[1] for s in _meta.samples], dtype=np.int32)
        del _meta
        # Pad to an exact multiple of step_size so every step has a full batch
        step_size = jax.device_count() * config.fid.device_batch_size
        pad_len = (step_size - len(val_labels_all) % step_size) % step_size
        val_labels_padded = np.pad(val_labels_all, (0, pad_len), constant_values=0)

        # ---- pmap sampling step with CFG ----
        p_sample_step = jax.pmap(
            partial(
                sample_step_cfg,
                model=model,
                rng_init=random.PRNGKey(config.sampling.seed),
                device_batch_size=config.fid.device_batch_size,
                config=config,
                cfg_scale=cfg_scale,
                num_classes=num_classes,
            ),
            axis_name='batch',
        )

        # Resume from partial latent checkpoint if available
        start_step = 0
        all_latents = []
        all_labels = []
        if resume and os.path.exists(partial_latents_path):
            data = np.load(partial_latents_path)
            completed = int(data['completed_steps'])
            all_latents = list(data['latents'])
            all_labels = list(data['labels'])
            start_step = completed
            pbar.update(start_step)
            log_for_0(f'Resuming generation from step {start_step}/{num_gen_steps}')

        for step in range(start_step, num_gen_steps):
            sample_idx = (jax.process_index() * jax.local_device_count()
                          + jnp.arange(jax.local_device_count()))
            sample_idx = jax.device_count() * step + sample_idx

            # Slice v2 labels for this step and distribute across local devices
            local_offset = jax.process_index() * jax.local_device_count() * config.fid.device_batch_size
            step_start = step * step_size + local_offset
            step_end = step_start + jax.local_device_count() * config.fid.device_batch_size
            labels_per_device = jnp.array(
                val_labels_padded[step_start:step_end].reshape(
                    jax.local_device_count(), config.fid.device_batch_size
                )
            )

            latent, labels_batch = p_sample_step(state, sample_idx=sample_idx,
                                                 labels=labels_per_device)
            latent = latent.reshape(-1, *latent.shape[2:])      # (N, 4, H, W)
            labels_batch = labels_batch.reshape(-1)              # (N,)
            all_latents.append(np.asarray(latent))
            all_labels.append(np.asarray(labels_batch))

            pbar.update(1)

            # Periodic checkpoint
            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                np.savez(
                    partial_latents_path,
                    latents=np.array(all_latents, dtype=object),
                    labels=np.array(all_labels, dtype=object),
                    completed_steps=step + 1,
                )

        all_latents = np.concatenate(all_latents, axis=0)[:len(val_labels_all)]
        all_labels = val_labels_all  # ordered v2 labels, no trimming needed

        # Save completed latents
        np.savez(latents_path, latents=all_latents, labels=all_labels)
        log_for_0(f'Phase 1 done: {all_latents.shape[0]} latents saved to {latents_path}')

        del state, p_sample_step, model
        gc.collect()

    # ==================================================================
    # Phase 2: Decode latents with VAE (model freed, VAE on GPU)
    # ==================================================================
    pbar.set_description('Decode')

    if skip_decode:
        log_for_0(f'Phase 2: SKIPPED (resuming from {decoded_path})')
        data = np.load(decoded_path)
        generated_images = data['images']
        log_for_0(f'  Loaded {generated_images.shape[0]} decoded samples.')
    else:
        log_for_0('Phase 2: Loading VAE for decoding...')
        latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size, image_size)

        num_decode_steps = int(np.ceil(len(all_latents) / decode_batch))
        if num_decode_steps != est_num_decode_steps and not skip_gen:
            pbar.total = num_gen_steps + num_decode_steps
            pbar.refresh()

        all_samples = []
        for i in range(0, len(all_latents), decode_batch):
            chunk = jnp.array(all_latents[i:i + decode_batch])
            actual_len = len(chunk)
            if actual_len < decode_batch:
                pad_shape = (decode_batch - actual_len,) + chunk.shape[1:]
                chunk = jnp.concatenate([chunk, jnp.zeros(pad_shape, dtype=chunk.dtype)], axis=0)
            decoded = latent_manager.decode(chunk)
            decoded = decoded[:actual_len]
            assert not jnp.any(jnp.isnan(decoded)), "NaN in decoded samples!"

            decoded = decoded.transpose(0, 2, 3, 1)
            decoded = 127.5 * decoded + 128.0
            decoded = jnp.clip(decoded, 0, 255).astype(jnp.uint8)
            all_samples.append(np.asarray(decoded))
            pbar.update(1)

        generated_images = np.concatenate(all_samples, axis=0)

        # Save decoded images for future resume
        np.savez(decoded_path, images=generated_images)
        log_for_0(f'Phase 2 done: {generated_images.shape[0]} images saved to {decoded_path}')

        del latent_manager, all_latents
        gc.collect()

    pbar.set_description('Metrics')
    pbar.close()

    # Free JAX-held GPU memory so PyTorch metrics (FID, IS, etc.) can use it.
    jax.clear_caches()
    try:
        backend = jax.lib.xla_bridge.get_backend()
        for buf in backend.live_buffers():
            buf.delete()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Phase 3: Load reference images
    # ==================================================================
    torch.cuda.is_available = _original_cuda_is_available

    log_for_0('Phase 3: Loading all real ImageNetV2 matched-frequency images for reference...')
    ref_dataset = ImageNetV2Dataset(
        data_path=data_path,
        image_size=config.dataset.raw_image_size,
        return_uint8=True,
    )
    class_names = ref_dataset.class_names
    real_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=200, shuffle=False, num_workers=4, drop_last=False
    )
    real_images = []
    real_labels_list = []
    for imgs, lbls in real_loader:
        if isinstance(imgs, torch.Tensor):
            real_images.append(imgs.numpy())
        else:
            real_images.append(np.stack([np.array(img) for img in imgs]))
        real_labels_list.append(np.array(lbls))
    real_images = np.concatenate(real_images, axis=0)
    real_labels = np.concatenate(real_labels_list, axis=0)
    log_for_0(f'Reference images: {real_images.shape}')

    # ==================================================================
    # Phase 4: Compute metrics (with per-metric caching on resume)
    # ==================================================================
    log_for_0('Phase 4: Computing metrics...')
    metrics, per_class_metrics = _compute_all_metrics(
        generated_images, all_labels, real_images, class_names,
        real_class_labels=real_labels,
        compute_fid_IS_per_class=FLAGS.compute_fid_IS_per_class,
        cache_dir=ckpt_dir if resume else None,
    )
    metrics.update({
        'num_samples': float(config.fid.num_samples),
        'num_steps': float(num_steps),
        'cfg_scale': cfg_scale,
    })

    # ==================================================================
    # Phase 5: Save final results and clean up
    # ==================================================================
    os.makedirs(plot_subdir, exist_ok=True)

    csv_path = os.path.join(plot_subdir, f'{subfolder_name}.csv')
    _save_metrics(metrics, csv_path)
    log_for_0(f'Metrics saved to {csv_path}')

    # Save sample grid
    png_path = eval_metrics.save_sample_grid(
        generated_images, output_dir, subfolder_name,
        labels=all_labels, class_names=class_names,
    )
    log_for_0(f'Sample grid saved to {png_path}')

    # Save per-class metrics CSV + density plot (in same subfolder as grid)
    per_class_int = {m: {int(k): v for k, v in d.items()} for m, d in per_class_metrics.items()}
    per_class_csv = os.path.join(plot_subdir, f'{subfolder_name}_per_class.csv')
    eval_metrics.save_per_class_metrics(per_class_int, per_class_csv, class_names)
    log_for_0(f'Per-class metrics saved to {per_class_csv}')

    density_path = os.path.join(plot_subdir, f'{subfolder_name}_per_class_density.png')
    eval_metrics.save_density_plots(per_class_int, density_path, class_names)
    log_for_0(f'Density plot saved to {density_path}')

    # ---- Summary ----
    log_for_0('=' * 60)
    log_for_0(f'FID:          {metrics["fid"]:.4f}')
    log_for_0(f'IS:           {metrics["is_mean"]:.4f} +/- {metrics["is_std"]:.4f}')
    log_for_0(f'CLIP Score:   {metrics["clip_score"]:.4f}')
    log_for_0(f'PickScore:    {metrics["pick_score"]:.4f}')
    log_for_0(f'CFG Scale:    {cfg_scale}')
    log_for_0(f'Steps:        {num_steps}')
    log_for_0(f'Output:       {csv_path}')
    log_for_0('=' * 60)

    return metrics


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    log_for_0('JAX process: %d / %d', jax.process_index(), jax.process_count())
    log_for_0('JAX local devices: %r', jax.local_devices())

    # Evaluate on the full ImageNetV2 matched-frequency set
    ds = ImageNetV2Dataset(data_path=FLAGS.data_path, return_uint8=True)
    num_samples = len(ds)
    del ds
    log_for_0(f'Evaluating with all {num_samples} ImageNetV2 matched-frequency samples')

    config = get_eval_config(
        checkpoint_path=FLAGS.checkpoint_path,
        model_cls=FLAGS.model_cls,
        vae_type=FLAGS.vae_type,
        num_samples=num_samples,
        device_batch_size=FLAGS.device_batch_size,
        seed=FLAGS.seed,
        num_steps=FLAGS.num_steps,
        omega=FLAGS.omega,
    )

    log_for_0('Config:\n{}'.format(config))

    evaluate(
        config,
        data_path=FLAGS.data_path,
        cfg_scale=FLAGS.omega,
        use_ema=not FLAGS.no_ema,
        resume=FLAGS.resume,
        checkpoint_every=FLAGS.checkpoint_every,
        model_name=FLAGS.model_name,
    )


if __name__ == '__main__':
    flags.mark_flags_as_required(['data_path', 'checkpoint_path'])
    app.run(main)
