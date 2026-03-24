"""Evaluate an iMeanFlow checkpoint on ImageNet.

Generates class-conditioned samples using CFG (with guidance interval) at
inference, decodes latents via VAE, and computes FID, Inception Score, CLIP
Score, and PickScore against real ImageNet images.

Uses a two-phase approach to avoid GPU OOM: first generates all latents
(model params on GPU), frees the model, then loads the VAE to decode.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
        --data_path=/path/to/imagenet \
        --checkpoint_path=path/to/checkpoint \
        --test \
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
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# Parse --num_gpus early (before JAX/PyTorch init) to set CUDA_VISIBLE_DEVICES.
def _parse_num_gpus():
    for arg in sys.argv[1:]:
        if arg.startswith('--num_gpus='):
            return int(arg.split('=', 1)[1])
    return 0

_NUM_GPUS = _parse_num_gpus()
if _NUM_GPUS > 0:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        allocated = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(allocated[:_NUM_GPUS])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(_NUM_GPUS))

from functools import partial
from pathlib import Path

import jax

if os.environ.get("COORDINATOR_ADDRESS") is not None:
    jax.distributed.initialize()

import jax.numpy as jnp
import numpy as np
import torch
# Keep PyTorch off GPU during generation phase
_original_cuda_is_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False
import torch.utils.data
from absl import app, flags, logging
from flax import jax_utils
from jax import random

from imf import iMeanFlow, generate
from utils.ckpt_util import restore_checkpoint
from utils.logging_util import log_for_0
from utils.lr_utils import lr_schedules
from utils.trainstate_util import create_train_state
from utils.vae_util import LatentManager

# Add src/ to import path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataloader import ImageNetDataset
from utils import eval as eval_metrics

warnings.filterwarnings("ignore")

# ======================================================================
# CLI Flags
# ======================================================================

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', None, 'Path to ImageNet root (contains train/ and/or val/)')
flags.DEFINE_string('checkpoint_path', None, 'Path to iMeanFlow checkpoint directory')
flags.DEFINE_string('model_str', 'MiT_B_2', 'MiT model variant')
flags.DEFINE_string('vae_type', 'mse', 'VAE variant (mse or ema)')
flags.DEFINE_integer('device_batch_size', 1, 'Per-device batch size for sampling')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_boolean('no_ema', False, 'Use raw params instead of EMA params')
flags.DEFINE_integer('num_steps', 1, 'Number of sampling steps')
flags.DEFINE_float('omega', 8.0, 'Guidance strength (cfg_scale)')
flags.DEFINE_float('t_min', 0.4, 'CFG interval lower bound')
flags.DEFINE_float('t_max', 0.65, 'CFG interval upper bound')
flags.DEFINE_boolean('test', False, '[unused] Always evaluates on the full val set')
flags.DEFINE_integer('num_gpus', 0, 'Number of GPUs to use (0 = use all available)')
flags.DEFINE_boolean('resume', False, 'Resume from checkpoint (skip completed phases)')
flags.DEFINE_integer('checkpoint_every', 50, 'Save latent checkpoint every N generation steps')
flags.DEFINE_string('model_name', 'imeanflow', 'Name used for output directory and file naming')
flags.DEFINE_boolean('compute_fid_IS_per_class', False,
                     'Compute per-class FID and IS (expensive — off by default)')
flags.DEFINE_string('output_dir', None,
                    'Override base output directory (default: outputs/imeanflow/steps_{num_steps})')


# ======================================================================
# Sampling (reuses generate() from imf.py via train.py's sample_step)
# ======================================================================

def sample_step(variable, sample_idx, labels, model, rng_init, device_batch_size,
                config, num_steps, omega, t_min, t_max):
    """Per-device sampling step with CFG. Used inside jax.pmap.

    labels: (device_batch_size,) int32 — val dataset class indices for this device.
    """
    rng_sample = random.fold_in(rng_init, sample_idx)

    images = generate(variable, model, rng_sample, device_batch_size,
                      config, num_steps, omega, t_min, t_max,
                      labels=labels)

    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW (for VAE decode)
    return images, labels


# ======================================================================
# Config
# ======================================================================

def get_eval_config(
    checkpoint_path,
    model_str='MiT_B_2',
    vae_type='mse',
    num_samples=50000,
    device_batch_size=1,
    seed=42,
    num_steps=1,
    omega=8.0,
    t_min=0.4,
    t_max=0.65,
):
    """Build evaluation config from defaults with CLI overrides."""
    from configs.default import get_config
    config = get_config()

    config.dataset.vae = vae_type

    config.model.model_str = model_str

    config.training.seed = seed

    config.sampling.num_steps = num_steps
    config.sampling.omega = omega
    config.sampling.t_min = t_min
    config.sampling.t_max = t_max

    config.fid.num_samples = num_samples
    config.fid.device_batch_size = device_batch_size

    config.load_from = checkpoint_path
    config.eval_only = True

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


def _save_best_worst_fid_grid(generated_images, all_labels, fid_per_class,
                               class_names, save_path):
    """Save a 2x10 image grid showing the 10 best and 10 worst per-class FID classes.

    Row 0 (top):    10 best classes — lowest per-class FID (green titles).
    Row 1 (bottom): 10 worst classes — highest per-class FID (red titles).
    Title format: "<class_name>\\n{fid:.2f}".
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_labels_arr = np.asarray(all_labels)
    sorted_classes = sorted(fid_per_class.items(), key=lambda x: x[1])
    best10  = sorted_classes[:10]   # lowest FID
    worst10 = sorted_classes[-10:]  # highest FID

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for row_idx, (group, color) in enumerate([(best10, '#1a9850'), (worst10, '#d73027')]):
        for col_idx, (cls, fid_val) in enumerate(group):
            ax = axes[row_idx, col_idx]
            match = np.where(all_labels_arr == cls)[0]
            if len(match):
                img = generated_images[match[0]]
            else:
                img = np.zeros((generated_images.shape[1], generated_images.shape[2], 3),
                               dtype=np.uint8)
            ax.imshow(img)
            ax.axis('off')
            name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            ax.set_title(f'{name}\n{fid_val:.2f}', fontsize=7, color=color, pad=2)

    axes[0, 0].set_ylabel('Best 10\n(↓ FID)', fontsize=9, color='#1a9850', labelpad=6)
    axes[1, 0].set_ylabel('Worst 10\n(↑ FID)', fontsize=9, color='#d73027', labelpad=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def _compute_all_metrics(generated_images, all_labels, real_images, class_names,
                         real_class_labels=None, compute_fid_IS_per_class=False,
                         cache_dir=None):
    """Compute all evaluation metrics and return as (global_dict, per_class_dict).

    When *cache_dir* is provided each metric is saved to a JSON file as
    soon as it finishes.  On a subsequent call with the same *cache_dir*,
    already-cached metrics are loaded instead of recomputed.
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
        # All unique classes expected in the generated set (used for both checks below)
        _expected_classes = set(int(c) for c in np.unique(all_labels))

        _cached_fid_classes = set(int(k) for k in cached.get('fid_per_class', {}).keys())
        if not _expected_classes.issubset(_cached_fid_classes):
            log_for_0('Computing per-class FID...')
            fid_pc = eval_metrics.compute_fid_per_class(
                real_images, real_class_labels, generated_images, all_labels
            )
            per_class['fid'] = {str(k): v for k, v in fid_pc.items()}
            _cache(results, per_class)
        else:
            log_for_0('FID per-class: cached')
            per_class['fid'] = {str(k): v for k, v in cached['fid_per_class'].items()}

        _cached_is_classes = set(int(k) for k in cached.get('is_per_class', {}).keys())
        if not _expected_classes.issubset(_cached_is_classes):
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


def evaluate(config, data_path, omega, t_min, t_max, use_ema, resume=False,
             checkpoint_every=50, model_name='imeanflow'):
    """Main evaluation: generate latents -> decode -> load reference -> metrics -> save."""
    from tqdm import tqdm

    image_size = config.dataset.image_size
    num_classes = config.dataset.num_classes
    num_steps = config.sampling.num_steps

    # ---- Output paths ----
    repo_root = Path(__file__).resolve().parent.parent.parent
    if FLAGS.output_dir:
        output_dir = FLAGS.output_dir if os.path.isabs(FLAGS.output_dir) else os.path.join(repo_root, FLAGS.output_dir)
    else:
        output_dir = os.path.join(repo_root, 'outputs', 'imeanflow', f'steps_{num_steps}')
    subfolder_name = f'{model_name}_steps_{num_steps}_imagenet'
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
        data = np.load(latents_path, allow_pickle=True)
        all_latents = data['latents']
        if all_latents.dtype == object:
            # Nested object arrays: use .tolist() to recursively unwrap to Python
            # lists, then re-stack as a proper numeric array.
            parts = [np.array(x.tolist() if isinstance(x, np.ndarray) else x,
                              dtype=np.float32) for x in all_latents]
            all_latents = np.stack(parts) if parts[0].ndim == 3 else np.concatenate(parts, axis=0)
        all_labels = data['labels']
        if all_labels.dtype == object:
            all_labels = np.concatenate([np.asarray(x) for x in all_labels], axis=0)
        log_for_0(f'  Loaded latents shape={all_latents.shape} dtype={all_latents.dtype}')
        log_for_0(f'  Loaded {all_latents.shape[0]} latent samples.')
    else:
        # ---- Model ----
        model_config = config.model.to_dict()
        model = iMeanFlow(**model_config, eval=True)

        # ---- Create train state and load checkpoint ----
        rng = random.key(0)
        lr_fn = lr_schedules(config, 1000)  # dummy steps_per_epoch
        state = create_train_state(rng, config, model, image_size, lr_fn)
        state = restore_checkpoint(state, config.load_from)

        if use_ema:
            params = state.ema_params
            log_for_0('Using EMA params.')
        else:
            params = state.params
            log_for_0('Using raw params.')
        del state

        params = jax.tree.map(lambda x: np.asarray(x), params)
        variable = jax_utils.replicate({"params": params})
        del params

        # ---- Load val labels — used to drive class-conditional generation ----
        _meta = ImageNetDataset(
            data_path=data_path, split='val',
            image_size=config.dataset.raw_image_size if hasattr(config.dataset, 'raw_image_size') else 256,
            return_uint8=True,
        )
        val_labels_all = np.array([s[1] for s in _meta.samples], dtype=np.int32)
        del _meta
        # Pad to an exact multiple of step_size so every step has a full batch
        step_size = jax.device_count() * config.fid.device_batch_size
        pad_len = (step_size - len(val_labels_all) % step_size) % step_size
        val_labels_padded = np.pad(val_labels_all, (0, pad_len), constant_values=0)

        # ---- pmap sampling step with CFG ----
        p_sample_step = jax.pmap(
            partial(
                sample_step,
                model=model,
                rng_init=random.PRNGKey(config.training.seed),
                device_batch_size=config.fid.device_batch_size,
                config=config,
                num_steps=config.sampling.num_steps,
                omega=omega,
                t_min=t_min,
                t_max=t_max,
            ),
            axis_name='batch',
        )

        # Resume from partial latent checkpoint if available
        start_step = 0
        all_latents = []
        all_labels = []
        if resume and os.path.exists(partial_latents_path):
            data = np.load(partial_latents_path, allow_pickle=True)
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

            # Slice val labels for this step and distribute across local devices
            local_offset = jax.process_index() * jax.local_device_count() * config.fid.device_batch_size
            step_start = step * step_size + local_offset
            step_end = step_start + jax.local_device_count() * config.fid.device_batch_size
            labels_per_device = jnp.array(
                val_labels_padded[step_start:step_end].reshape(
                    jax.local_device_count(), config.fid.device_batch_size
                )
            )

            latent, labels_batch = p_sample_step(variable, sample_idx=sample_idx,
                                                 labels=labels_per_device)
            latent = latent.reshape(-1, *latent.shape[2:])  # (N, 4, H, W)
            labels_batch = labels_batch.reshape(-1)          # (N,)

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
        all_labels = val_labels_all  # ordered val labels, no trimming needed

        # Save completed latents
        np.savez(latents_path, latents=all_latents, labels=all_labels)
        log_for_0(f'Phase 1 done: {all_latents.shape[0]} latents saved to {latents_path}')

        del variable, p_sample_step, model
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

    # Free JAX-held GPU memory so PyTorch metrics can use it.
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

    log_for_0('Phase 3: Loading all real ImageNet val images for reference...')
    ref_dataset = ImageNetDataset(
        data_path=data_path,
        split='val',
        image_size=config.dataset.raw_image_size if hasattr(config.dataset, 'raw_image_size') else 256,
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
        cache_dir=ckpt_dir,          # always cache; per-class metrics are expensive
    )
    metrics.update({
        'num_samples': float(config.fid.num_samples),
        'num_steps': float(num_steps),
        'cfg_scale': omega,
        't_min': t_min,
        't_max': t_max,
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

    if per_class_metrics.get('fid'):
        fid_grid_path = os.path.join(plot_subdir, f'{subfolder_name}_best_worst_fid.png')
        _save_best_worst_fid_grid(
            generated_images, all_labels,
            {int(k): v for k, v in per_class_metrics['fid'].items()},
            class_names, fid_grid_path,
        )
        log_for_0(f'Best/worst FID grid saved to {fid_grid_path}')

    # ---- Summary ----
    log_for_0('=' * 60)
    log_for_0(f'FID:          {metrics["fid"]:.4f}')
    log_for_0(f'IS:           {metrics["is_mean"]:.4f} +/- {metrics["is_std"]:.4f}')
    log_for_0(f'CLIP Score:   {metrics["clip_score"]:.4f}')
    log_for_0(f'PickScore:    {metrics["pick_score"]:.4f}')
    log_for_0(f'CFG Scale:    {omega}')
    log_for_0(f't_min:        {t_min}')
    log_for_0(f't_max:        {t_max}')
    log_for_0(f'Steps:        {num_steps}')
    log_for_0(f'Output:       {csv_path}')
    log_for_0('=' * 60)

    return metrics


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    log_for_0('JAX process: %d / %d', jax.process_index(), jax.process_count())
    log_for_0('JAX local devices: %r', jax.local_devices())

    # Always evaluate on the full val set
    ds = ImageNetDataset(data_path=FLAGS.data_path, split='val', return_uint8=True)
    num_samples = len(ds)
    del ds
    log_for_0(f'Evaluating with all {num_samples} ImageNet val samples')

    config = get_eval_config(
        checkpoint_path=FLAGS.checkpoint_path,
        model_str=FLAGS.model_str,
        vae_type=FLAGS.vae_type,
        num_samples=num_samples,
        device_batch_size=FLAGS.device_batch_size,
        seed=FLAGS.seed,
        num_steps=FLAGS.num_steps,
        omega=FLAGS.omega,
        t_min=FLAGS.t_min,
        t_max=FLAGS.t_max,
    )

    log_for_0('Config:\n{}'.format(config))

    evaluate(
        config,
        data_path=FLAGS.data_path,
        omega=FLAGS.omega,
        t_min=FLAGS.t_min,
        t_max=FLAGS.t_max,
        use_ema=not FLAGS.no_ema,
        resume=FLAGS.resume,
        checkpoint_every=FLAGS.checkpoint_every,
        model_name=FLAGS.model_name,
    )


if __name__ == '__main__':
    flags.mark_flags_as_required(['data_path', 'checkpoint_path'])
    app.run(main)
