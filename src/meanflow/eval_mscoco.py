"""Evaluate a MeanFlow checkpoint on MS-COCO with classifier-free guidance.

Generates class-conditioned samples using CFG at inference time, decodes
latents via VAE, and computes FID, Inception Score, CLIP Score, and PickScore.

Uses a two-phase approach to avoid GPU OOM: first generates all latents
(model params on GPU), frees the model, then loads the VAE to decode.

Usage:
    python eval_mscoco.py \
        --data_path=/path/to/coco/images \
        --data_mapping=data/mappings/coco.json \
        --ckpt_path=weights/checkpoint_1200960/ \
        --test \
        --num_steps=1 \
        --cfg_scale=1.5
"""
import csv
import gc
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Flax classes are deprecated.*")
os.environ.setdefault('CUDA_VISIBLE_DEVICES_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
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
from utils.dataloader import MSCOCODataset
from utils import eval as eval_metrics

supress_checkpt_info()
warnings.filterwarnings("ignore")

# ======================================================================
# CLI Flags
# ======================================================================

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', None, 'Path to COCO images directory (e.g., coco/images/)')
flags.DEFINE_string('data_mapping', None, 'Path to mapping JSON from compute_map.py (e.g., data/mappings/coco.json)')
flags.DEFINE_string('ckpt_path', None, 'Path to MeanFlow checkpoint directory')
flags.DEFINE_string('model_cls', 'DiT_B_4', 'DiT model variant')
flags.DEFINE_string('vae_type', 'mse', 'VAE variant (mse or ema)')
flags.DEFINE_integer('device_batch_size', 1, 'Per-device batch size for sampling')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_integer('num_steps', 1, 'Number of ODE sampling steps (1 for one-step, >1 for multi-step)')
flags.DEFINE_float('cfg_scale', 1.5, 'Classifier-free guidance scale (1.0 = no guidance)')
flags.DEFINE_boolean('no_ema', False, 'Use raw params instead of EMA params')
flags.DEFINE_boolean('test', False, 'Evaluate on 50,000 samples (otherwise use full dataset)')


# ======================================================================
# CFG Generation
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


# ======================================================================
# Sampling Step (for pmap)
# ======================================================================

def sample_step_cfg(variable, sample_idx, model, rng_init, device_batch_size,
                    config, cfg_scale, num_classes):
    """Per-device sampling step with CFG. Used inside jax.pmap."""
    rng_sample = random.fold_in(rng_init, sample_idx)
    rng_sample, rng_labels = jax.random.split(rng_sample)

    labels = jax.random.randint(rng_labels, (device_batch_size,), 0, num_classes)

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
    ckpt_path,
    model_cls='DiT_B_4',
    vae_type='mse',
    num_samples=50000,
    device_batch_size=1,
    seed=42,
    num_steps=1,
):
    """Build evaluation config."""
    config = ml_collections.ConfigDict()

    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.name = 'mscoco'
    dataset.image_size = 32        # latent space (256 // 8)
    dataset.raw_image_size = 256
    dataset.image_channels = 4
    dataset.num_classes = 1000     # match pretrained embedding size
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
    method.omega = 1.0
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

    config.load_from = ckpt_path

    return config


# ======================================================================
# Main Evaluation
# ======================================================================

def evaluate(config, data_path, data_mapping, cfg_scale, use_ema):
    """Main evaluation: generate with CFG -> decode -> compute metrics -> save."""
    image_size = config.dataset.image_size
    num_classes = config.dataset.num_classes
    num_steps = config.sampling.num_steps

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

    # ==================================================================
    # Phase 1: Generate all latents with CFG (model on GPU)
    # ==================================================================
    num_gen_steps = int(np.ceil(
        config.fid.num_samples / config.fid.device_batch_size / jax.device_count()
    ))
    log_for_0(f'Phase 1: Generating latents with CFG (scale={cfg_scale}, '
              f'{num_steps}-step, {num_gen_steps} batches)...')

    all_latents = []
    all_labels = []
    for step in range(num_gen_steps):
        sample_idx = (jax.process_index() * jax.local_device_count()
                      + jnp.arange(jax.local_device_count()))
        sample_idx = jax.device_count() * step + sample_idx

        if step == 0:
            log_for_0('Note: the first step may be significantly slower (compilation)')
        log_for_0(f'  Sampling step {step + 1} / {num_gen_steps}...')

        latent, labels_batch = p_sample_step(state, sample_idx=sample_idx)
        latent = latent.reshape(-1, *latent.shape[2:])      # (N, 4, H, W)
        labels_batch = labels_batch.reshape(-1)              # (N,)
        all_latents.append(np.asarray(latent))
        all_labels.append(np.asarray(labels_batch))

    all_latents = np.concatenate(all_latents, axis=0)[:config.fid.num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:config.fid.num_samples]
    log_for_0(f'Phase 1 done: {all_latents.shape[0]} latent samples on CPU.')

    del state, p_sample_step, model
    gc.collect()

    # ==================================================================
    # Phase 2: Decode latents with VAE (VAE on GPU, model freed)
    # ==================================================================
    log_for_0('Phase 2: Loading VAE for decoding...')
    latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size, image_size)

    decode_batch = config.fid.device_batch_size * jax.local_device_count()
    all_samples = []
    num_decode_steps = int(np.ceil(len(all_latents) / decode_batch))

    for i in range(0, len(all_latents), decode_batch):
        chunk = jnp.array(all_latents[i:i + decode_batch])
        decoded = latent_manager.decode(chunk)
        assert not jnp.any(jnp.isnan(decoded)), "NaN in decoded samples!"

        decoded = decoded.transpose(0, 2, 3, 1)
        decoded = 127.5 * decoded + 128.0
        decoded = jnp.clip(decoded, 0, 255).astype(jnp.uint8)
        all_samples.append(np.asarray(decoded))

        step_num = i // decode_batch + 1
        log_for_0(f'  Decode step {step_num} / {num_decode_steps}')

    generated_images = np.concatenate(all_samples, axis=0)
    log_for_0(f'Phase 2 done: {generated_images.shape[0]} decoded samples.')

    del latent_manager, all_latents
    gc.collect()

    # ==================================================================
    # Phase 3: Collect real COCO images for reference
    # ==================================================================
    log_for_0('Phase 3: Collecting real images for reference...')

    # Restore torch GPU access for metrics computation
    torch.cuda.is_available = _original_cuda_is_available

    real_dataset = MSCOCODataset(
        data_path=data_path,
        data_mapping=data_mapping,
        image_size=config.dataset.raw_image_size,
        return_uint8=True,
    )
    class_names = real_dataset.class_names

    rng_np = np.random.RandomState(config.training.seed)
    real_indices = rng_np.choice(
        len(real_dataset),
        size=min(config.fid.num_samples, len(real_dataset)),
        replace=False,
    )
    real_subset = torch.utils.data.Subset(real_dataset, real_indices)
    real_loader = torch.utils.data.DataLoader(
        real_subset, batch_size=200, shuffle=False, num_workers=4, drop_last=False
    )

    real_images = []
    for imgs, _ in real_loader:
        if isinstance(imgs, torch.Tensor):
            real_images.append(imgs.numpy())
        else:
            real_images.append(np.stack([np.array(img) for img in imgs]))
    real_images = np.concatenate(real_images, axis=0)
    log_for_0(f'Reference images: {real_images.shape}')

    # ==================================================================
    # Phase 4: Compute Metrics
    # ==================================================================
    log_for_0('Phase 4: Computing metrics...')

    log_for_0('Computing FID...')
    fid_score = eval_metrics.compute_fid(real_images, generated_images)
    log_for_0(f'FID: {fid_score:.4f}')

    log_for_0('Computing Inception Score...')
    is_mean, is_std = eval_metrics.compute_inception_score(generated_images)
    log_for_0(f'IS: {is_mean:.4f} +/- {is_std:.4f}')

    log_for_0('Computing CLIP Score...')
    clip_score = eval_metrics.compute_clip_score(
        generated_images, all_labels, class_names
    )
    log_for_0(f'CLIP Score: {clip_score:.4f}')

    prompts = [f"a photo of a {class_names[l]}" for l in all_labels]

    log_for_0('Computing PickScore...')
    pick_score = eval_metrics.compute_pick_score(generated_images, prompts)
    log_for_0(f'PickScore: {pick_score:.4f}')

    # ==================================================================
    # Phase 5: Save Results
    # ==================================================================
    log_for_0('Phase 5: Saving results...')

    output_dir = os.path.join('outputs', 'meanflow', str(num_steps))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'meanflow_{num_steps}_mscoco.csv')

    metrics = {
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std,
        'clip_score': clip_score,
        'pick_score': pick_score,
        'num_samples': float(config.fid.num_samples),
        'num_steps': float(num_steps),
        'cfg_scale': cfg_scale,
    }

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    log_for_0(f'Metrics saved to {csv_path}')

    # Save sample grid
    sample_images = generated_images[:10]
    png_path = eval_metrics.save_sample_grid(
        sample_images, output_dir, f'meanflow_{num_steps}_mscoco'
    )
    log_for_0(f'Sample grid saved to {png_path}')

    # ---- Summary ----
    log_for_0('=' * 60)
    log_for_0(f'FID:          {fid_score:.4f}')
    log_for_0(f'IS:           {is_mean:.4f} +/- {is_std:.4f}')
    log_for_0(f'CLIP Score:   {clip_score:.4f}')
    log_for_0(f'PickScore:    {pick_score:.4f}')
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

    # Determine number of samples
    if FLAGS.test:
        num_samples = 50000
    else:
        ds = MSCOCODataset(
            data_path=FLAGS.data_path,
            data_mapping=FLAGS.data_mapping,
            image_size=256,
            return_uint8=True,
        )
        num_samples = len(ds)
        del ds
    log_for_0(f'Evaluating with {num_samples} samples (--test={FLAGS.test})')

    config = get_eval_config(
        ckpt_path=FLAGS.ckpt_path,
        model_cls=FLAGS.model_cls,
        vae_type=FLAGS.vae_type,
        num_samples=num_samples,
        device_batch_size=FLAGS.device_batch_size,
        seed=FLAGS.seed,
        num_steps=FLAGS.num_steps,
    )

    log_for_0('Config:\n{}'.format(config))

    evaluate(
        config,
        data_path=FLAGS.data_path,
        data_mapping=FLAGS.data_mapping,
        cfg_scale=FLAGS.cfg_scale,
        use_ema=not FLAGS.no_ema,
    )


if __name__ == '__main__':
    flags.mark_flags_as_required(['data_path', 'data_mapping', 'ckpt_path'])
    app.run(main)
