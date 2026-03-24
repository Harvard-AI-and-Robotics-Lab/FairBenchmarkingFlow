"""Evaluate RAE (DiTDH) on reLAIONet (CLIP-filtered web-scraped ImageNet subset).

Generates class-conditioned samples using the pretrained Stage 2 diffusion
model, decodes latents via the Stage 1 RAE decoder, and computes FID,
Inception Score, CLIP Score, PickScore, and ImageReward against real
reLAIONet images using the shared evaluation utilities.

Supports multi-GPU generation via PyTorch DDP (torchrun), with metrics
computed on rank 0 after gathering all generated images.

Follows the same phase structure as RAE's eval_imagenet.py:
    Phase 1: Generate images across GPUs (RAE-specific, DDP)
    Phase 2: Gather to rank 0, load real reLAIONet references (shared dataloader)
    Phase 3: Compute metrics on rank 0 (shared eval)
    Phase 4: Save results (standardized output format)

Usage:
    # 5-sample sanity check (single GPU is fine)
    torchrun --standalone --nproc_per_node=1 eval_relaionet.py \
        --config configs/stage2/sampling/<config>.yaml \
        --data_path /path/to/data \
        --num_samples 5

    # Full run on 4 GPUs
    torchrun --standalone --nproc_per_node=4 eval_relaionet.py \
        --config configs/stage2/sampling/<config>.yaml \
        --data_path /path/to/data
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.cuda.amp import autocast


# ======================================================================
# Path Setup (identical to RAE eval_imagenet.py)
# ======================================================================

_script_dir = Path(__file__).resolve().parent
_rae_src = str(_script_dir / 'src')     # src/rae/src/  (RAE's code)
_shared_src = str(_script_dir.parent)   # src/          (shared utils)

sys.path.insert(0, _rae_src)

from omegaconf import OmegaConf
from stage1 import RAE
from stage2.transport import create_transport, Sampler
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs

for _mod_name in list(sys.modules):
    if _mod_name == 'utils' or _mod_name.startswith('utils.'):
        del sys.modules[_mod_name]

sys.path.insert(0, _shared_src)

from utils.dataloader import ReLAIONetDataset
from utils import eval as eval_metrics

if _rae_src not in sys.path:
    sys.path.append(_rae_src)

warnings.filterwarnings("ignore")


# ======================================================================
# Helpers
# ======================================================================

def _get_num_steps_from_config(config_path, num_steps_override):
    """Parse the RAE sampling config to determine num_steps before generation."""
    if num_steps_override is not None:
        return num_steps_override
    try:
        cfg = OmegaConf.load(config_path)
        _, _, _, sampler_config, _, _, _, _ = parse_configs(cfg)
        if sampler_config is not None:
            params = dict(sampler_config.get("params", {}))
            v = params.get("num_steps", params.get("steps"))
            if v is not None:
                return int(v)
    except Exception:
        pass
    return "unknown"


def _save_best_worst_fid_grid(generated_images, all_labels, fid_per_class,
                               class_names, save_path):
    """Save a 2x10 image grid showing the 10 best and 10 worst per-class FID classes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_labels_arr = np.asarray(all_labels)
    sorted_classes = sorted(fid_per_class.items(), key=lambda x: x[1])
    best10  = sorted_classes[:10]
    worst10 = sorted_classes[-10:]

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
        print('Computing FID...')
        results['fid'] = float(eval_metrics.compute_fid(real_images, generated_images))
        _cache(results, per_class)
    else:
        print(f'FID: cached ({results["fid"]:.4f})')

    if 'is_mean' not in results:
        print('Computing Inception Score...')
        is_mean, is_std = eval_metrics.compute_inception_score(generated_images)
        results['is_mean'] = float(is_mean)
        results['is_std'] = float(is_std)
        _cache(results, per_class)
    else:
        print(f'IS: cached ({results["is_mean"]:.4f})')

    if compute_fid_IS_per_class and real_class_labels is not None:
        _expected_classes = set(int(c) for c in np.unique(all_labels))

        _cached_fid_classes = set(int(k) for k in cached.get('fid_per_class', {}).keys())
        if not _expected_classes.issubset(_cached_fid_classes):
            print('Computing per-class FID...')
            fid_pc = eval_metrics.compute_fid_per_class(
                real_images, real_class_labels, generated_images, all_labels
            )
            per_class['fid'] = {str(k): v for k, v in fid_pc.items()}
            _cache(results, per_class)
        else:
            print('FID per-class: cached')
            per_class['fid'] = {str(k): v for k, v in cached['fid_per_class'].items()}

        _cached_is_classes = set(int(k) for k in cached.get('is_per_class', {}).keys())
        if not _expected_classes.issubset(_cached_is_classes):
            print('Computing per-class IS...')
            is_pc = eval_metrics.compute_inception_score_per_class(
                generated_images, all_labels
            )
            per_class['inception_score'] = {str(k): v for k, v in is_pc.items()}
            _cache(results, per_class)
        else:
            print('IS per-class: cached')
            per_class['inception_score'] = {str(k): v for k, v in cached['is_per_class'].items()}

    if 'clip_score' not in results:
        print('Computing CLIP Score...')
        clip_mean, clip_per_class = eval_metrics.compute_clip_score(
            generated_images, all_labels, class_names, return_per_class=True
        )
        results['clip_score'] = float(clip_mean)
        per_class['clip_score'] = {str(k): v for k, v in clip_per_class.items()}
        _cache(results, per_class)
    else:
        print(f'CLIP Score: cached ({results["clip_score"]:.4f})')
        per_class['clip_score'] = {str(k): v for k, v in cached.get('clip_score_per_class', {}).items()}

    prompts = [f"a photo of a {class_names[l]}" for l in all_labels]

    if 'pick_score' not in results:
        print('Computing PickScore...')
        pick_mean, pick_per_class = eval_metrics.compute_pick_score(
            generated_images, prompts, class_labels=all_labels, return_per_class=True
        )
        results['pick_score'] = float(pick_mean)
        per_class['pick_score'] = {str(k): v for k, v in pick_per_class.items()}
        _cache(results, per_class)
    else:
        print(f'PickScore: cached ({results["pick_score"]:.4f})')
        per_class['pick_score'] = {str(k): v for k, v in cached.get('pick_score_per_class', {}).items()}

    return results, per_class


# ======================================================================
# DDP Helpers
# ======================================================================

def setup_ddp():
    """Initialize DDP and return (rank, world_size, device)."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)
    return rank, world_size, device


def cleanup_ddp():
    """Destroy the DDP process group."""
    dist.destroy_process_group()


def log(rank, msg):
    """Print only from rank 0."""
    if rank == 0:
        print(msg)


# ======================================================================
# Phase 1: Image Generation (RAE-specific, DDP-parallelized)
# ======================================================================

def generate_images(config_path, num_samples, batch_size, seed, precision,
                    rank, world_size, device, val_labels_rank,
                    num_steps_override=None, cfg_scale_override=None, ckpt_every=100,
                    partial_ckpt_dir=None):
    """Generate images using RAE's two-stage pipeline (one GPU's share)."""
    # ---- Parse config ----
    cfg = OmegaConf.load(config_path)
    (rae_config, model_config, transport_config, sampler_config,
     guidance_config, misc, _, _) = parse_configs(cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 entries.")

    misc = {} if misc is None else dict(misc)

    # ---- Latent space setup ----
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    log(rank, f"Latent size: {latent_size}, time_dist_shift: {time_dist_shift:.4f}")

    # ---- Load models ----
    use_bf16 = precision == "bf16"
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    log(rank, "Loading RAE (Stage 1 decoder) and diffusion model (Stage 2)...")
    rae: RAE = instantiate_from_config(rae_config).to(device)
    model = instantiate_from_config(model_config).to(device)
    rae.eval()
    model.eval()

    # ---- Create transport and sampler ----
    transport_params = {}
    if transport_config is not None:
        transport_params = dict(transport_config.get("params", {}))
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)

    sampler_config = {} if sampler_config is None else dict(sampler_config)
    sampler_mode = sampler_config.get("mode", "ODE")
    sampler_params = dict(sampler_config.get("params", {}))
    if num_steps_override is not None:
        sampler_params["num_steps"] = num_steps_override + 1  # +1: integrator needs N+1 time points for N steps

    mode = sampler_mode.upper()
    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode: {sampler_mode}")

    # ---- Guidance setup ----
    guidance_config = {} if guidance_config is None else dict(guidance_config)
    guidance_scale = guidance_config.get("scale", 1.0)
    if cfg_scale_override is not None:
        guidance_scale = cfg_scale_override
    guidance_method = guidance_config.get("method", "cfg")
    t_min = guidance_config.get("t_min", guidance_config.get("t-min", 0.0))
    t_max = guidance_config.get("t_max", guidance_config.get("t-max", 1.0))

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guid_model_config = guidance_config.get("guidance_model")
        if guid_model_config is None:
            raise ValueError("Autoguidance requires a guidance_model in config.")
        guid_model = instantiate_from_config(guid_model_config).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    using_cfg = guidance_scale > 1.0

    # Report the user-facing step count (not the integrator's internal point count)
    num_steps = (num_steps_override if num_steps_override is not None
                 else sampler_params.get("num_steps", sampler_params.get("steps", "unknown")))

    # ---- This rank's share comes from the contiguous reLAIONet label slice ----
    samples_per_rank = len(val_labels_rank)

    log(rank, f"Generating {num_samples} total samples across {world_size} GPU(s) "
              f"({samples_per_rank} per GPU, guidance={guidance_method}, "
              f"scale={guidance_scale}, steps={num_steps}, mode={mode})")

    # ---- Generation loop ----
    rank_seed = seed + rank
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    local_images = []
    local_labels = []
    num_generated = 0
    batch_count = 0

    # ---- Check for partial checkpoint from a previous crashed run ----
    partial_dir = (Path(partial_ckpt_dir) if partial_ckpt_dir is not None
                   else Path(__file__).resolve().parent / 'outputs') / '_gen_partial'
    partial_path = partial_dir / f'rank_{rank}_partial.npz'
    if partial_path.exists():
        log(rank, f"  Rank {rank}: Found partial checkpoint, resuming...")
        partial_data = np.load(str(partial_path))
        partial_images = partial_data['images']
        partial_labels = partial_data['labels']
        num_generated = int(partial_data['num_generated'])
        local_images.append(partial_images)
        local_labels.append(partial_labels)
        del partial_data, partial_images, partial_labels
        log(rank, f"  Rank {rank}: Resumed from {num_generated} / {samples_per_rank}")
        torch.manual_seed(rank_seed + num_generated)
        torch.cuda.manual_seed(rank_seed + num_generated)

    with torch.inference_mode():
        while num_generated < samples_per_rank:
            current_batch = min(batch_size, samples_per_rank - num_generated)

            with autocast(**autocast_kwargs):
                z = torch.randn(current_batch, *latent_size, device=device)

                y = torch.from_numpy(
                    val_labels_rank[num_generated:num_generated + current_batch].copy()
                ).to(device, dtype=torch.long)

                model_kwargs = dict(y=y)
                model_fn = model.forward

                if using_cfg:
                    z = torch.cat([z, z], dim=0)
                    y_null = torch.full((current_batch,), null_label, device=device)
                    y_cfg = torch.cat([y, y_null], dim=0)
                    model_kwargs = dict(
                        y=y_cfg,
                        cfg_scale=guidance_scale,
                        cfg_interval=(t_min, t_max),
                    )
                    if guidance_method == "autoguidance":
                        if guid_model_forward is None:
                            raise RuntimeError("Guidance model not initialized.")
                        model_kwargs["additional_model_forward"] = guid_model_forward
                        model_fn = model.forward_with_autoguidance
                    else:
                        model_fn = model.forward_with_cfg

                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)

                samples = rae.decode(samples).clamp(0, 1)
                samples_np = (samples
                              .mul(255)
                              .permute(0, 2, 3, 1)
                              .to("cpu", dtype=torch.uint8)
                              .numpy())

            local_images.append(samples_np)
            local_labels.append(y.cpu().numpy())
            num_generated += current_batch
            batch_count += 1
            log(rank, f"  Rank {rank}: generated {num_generated} / {samples_per_rank}")

            if ckpt_every > 0 and batch_count % ckpt_every == 0:
                partial_dir.mkdir(parents=True, exist_ok=True)
                _tmp_imgs = np.concatenate(local_images, axis=0)
                _tmp_lbls = np.concatenate(local_labels, axis=0)
                np.savez(str(partial_path),
                         images=_tmp_imgs, labels=_tmp_lbls,
                         num_generated=np.array(num_generated))
                del _tmp_imgs, _tmp_lbls
                log(rank, f"  Rank {rank}: saved partial checkpoint ({num_generated} images)")

    local_images = np.concatenate(local_images, axis=0)[:samples_per_rank]
    local_labels = np.concatenate(local_labels, axis=0)[:samples_per_rank]

    del rae, model, sample_fn, sampler, transport
    if guid_model_forward is not None:
        del guid_model
    gc.collect()
    torch.cuda.empty_cache()

    return local_images, local_labels, num_steps, guidance_scale


def gather_arrays(local_array, rank, world_size):
    """Gather numpy arrays from all ranks to rank 0 via disk."""
    tmp_dir = Path(__file__).resolve().parent / 'outputs' / '_gather_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f'rank_{rank}.npy'

    np.save(str(tmp_path), local_array)
    dist.barrier()

    if rank == 0:
        arrays = []
        for r in range(world_size):
            rpath = tmp_dir / f'rank_{r}.npy'
            arrays.append(np.load(str(rpath)))
            rpath.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        return np.concatenate(arrays, axis=0)

    dist.barrier()
    return None


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAE on reLAIONet")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to RAE sampling config YAML '
                             '(e.g., configs/stage2/sampling/...)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to reLAIONet root (contains reLAIONet-cleaned-v1/)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Override number of samples to generate '
                             '(use for sanity checks, e.g., --num_samples 5)')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Per-GPU per-step batch size for generation '
                             '(lower = less VRAM; default 25)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'bf16'],
                        help='Inference precision')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image resolution for reference images')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Override number of sampling steps from config')
    parser.add_argument('--cfg_scale', type=float, default=None,
                        help='Override CFG guidance scale from config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to generated_checkpoint.npz to skip generation')
    parser.add_argument('--output_suffix', type=str, default=None,
                        help='Suffix for output folder name '
                             '(e.g., "AG" -> steps_25_AG)')
    parser.add_argument('--ckpt_every', type=int, default=100,
                        help='Save partial generation checkpoint every N batches '
                             'per rank (default 100). Set 0 to disable.')
    parser.add_argument('--model_name', type=str, default='rae',
                        help='Name used for output CSV and PNG tag '
                             '(e.g. "rae" or "natural_rae").')
    parser.add_argument('--compute_fid_IS_per_class', action='store_true', default=False,
                        help='Compute per-class FID and IS (expensive — off by default)')
    args = parser.parse_args()

    # ---- DDP setup ----
    rank, world_size, device = setup_ddp()
    log(rank, f"DDP initialized: rank {rank}, world_size {world_size}, "
              f"device {device}")

    # ---- Output paths (computed early so all ranks share consistent paths) ----
    repo_root = Path(__file__).resolve().parent.parent.parent
    _num_steps_early = _get_num_steps_from_config(args.config, args.num_steps)
    folder_name = f'steps_{_num_steps_early}'
    if args.output_suffix:
        folder_name += f'_{args.output_suffix}'
    tag = args.model_name
    if args.output_suffix:
        tag += f'_{args.output_suffix}'
    subfolder_name = f'{tag}_steps_{_num_steps_early}_relaionet'
    output_dir = repo_root / 'outputs' / 'rae' / folder_name
    plot_subdir = output_dir / subfolder_name
    ckpt_dir = plot_subdir / '_checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'generated_checkpoint.npz'

    # ---- Determine number of samples ----
    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        if rank == 0:
            ds = ReLAIONetDataset(data_path=args.data_path,
                                  image_size=args.image_size, return_uint8=True)
            num_samples = len(ds)
            del ds
        else:
            num_samples = 0
        num_samples_tensor = torch.tensor([num_samples], dtype=torch.long, device="cuda")
        dist.broadcast(num_samples_tensor, src=0)
        num_samples = int(num_samples_tensor.item())

    log(rank, f"Evaluating with {num_samples} samples on {world_size} GPU(s)")

    # ---- Load reLAIONet labels on all ranks and distribute contiguous slices ----
    _meta = ReLAIONetDataset(data_path=args.data_path,
                             image_size=args.image_size, return_uint8=True)
    val_labels_all = np.array([s[1] for s in _meta.samples], dtype=np.int32)
    del _meta
    n_per_rank = (num_samples + world_size - 1) // world_size
    rank_start = rank * n_per_rank
    rank_end = min(rank_start + n_per_rank, num_samples)
    val_labels_rank = val_labels_all[rank_start:rank_end]

    # ==================================================================
    # Phase 1: Generate images (or resume from checkpoint)
    # ==================================================================
    if args.resume and rank == 0:
        log(rank, f"Resuming from checkpoint: {args.resume}")
        ckpt = np.load(args.resume)
        generated_images = ckpt["images"][:num_samples]
        all_labels = ckpt["labels"][:num_samples]
        del ckpt
        num_steps = args.num_steps if args.num_steps is not None else "unknown"
        guidance_scale = args.cfg_scale if args.cfg_scale is not None else 1.0
        log(rank, f"Loaded {generated_images.shape[0]} images from checkpoint.")
    elif args.resume:
        generated_images = None
        all_labels = None
        num_steps = args.num_steps if args.num_steps is not None else "unknown"
        guidance_scale = args.cfg_scale if args.cfg_scale is not None else 1.0
    else:
        log(rank, "=" * 60)
        log(rank, "Phase 1: Generating images with RAE...")
        log(rank, "=" * 60)

        local_images, local_labels, num_steps, guidance_scale = generate_images(
            config_path=args.config,
            num_samples=num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            precision=args.precision,
            rank=rank,
            world_size=world_size,
            device=device,
            val_labels_rank=val_labels_rank,
            num_steps_override=args.num_steps,
            cfg_scale_override=args.cfg_scale,
            ckpt_every=args.ckpt_every,
            partial_ckpt_dir=str(ckpt_dir),
        )
        log(rank, f"Rank {rank} generated {local_images.shape[0]} images locally")

        log(rank, "Gathering generated images from all ranks to rank 0...")
        tmp_dir = ckpt_dir / '_gather_tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.savez(str(tmp_dir / f'rank_{rank}.npz'),
                 images=local_images, labels=local_labels)

        del local_images, local_labels
        gc.collect()
        torch.cuda.empty_cache()

        dist.barrier()

        if rank == 0:
            all_images_list, all_labels_list = [], []
            for r in range(world_size):
                rpath = tmp_dir / f'rank_{r}.npz'
                data = np.load(str(rpath))
                all_images_list.append(data['images'])
                all_labels_list.append(data['labels'])
            generated_images = np.concatenate(all_images_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
        else:
            generated_images = None
            all_labels = None

        dist.barrier()

    dist.barrier()
    num_gpus_total = world_size
    cleanup_ddp()

    if rank != 0:
        return

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.set_device(0)

    num_visible_gpus = torch.cuda.device_count()
    print(f"Rank 0 now sees {num_visible_gpus} GPU(s) for metrics "
          f"(eval.py will use DataParallel)")

    generated_images = generated_images[:num_samples]
    all_labels = all_labels[:num_samples]
    print(f"Phase 1 done: {generated_images.shape} generated "
          f"(dtype={generated_images.dtype})")

    print(f"Saving generation checkpoint to {ckpt_path}...")
    np.savez_compressed(str(ckpt_path),
                        images=generated_images,
                        labels=all_labels)
    print(f"Checkpoint saved ({generated_images.shape[0]} images).")

    # ==============================================================
    # Phase 2: Load real reLAIONet reference images
    # ==============================================================
    print("=" * 60)
    print("Phase 2: Loading real reLAIONet reference images...")
    print("=" * 60)

    ref_dataset = ReLAIONetDataset(
        data_path=args.data_path,
        image_size=args.image_size,
        return_uint8=True,
    )
    class_names = ref_dataset.class_names

    real_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=200, shuffle=False,
        num_workers=4, drop_last=False,
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
    print(f"Reference images: {real_images.shape}")

    # ==============================================================
    # Phase 3: Compute metrics (with per-metric caching on resume)
    # ==============================================================
    print("=" * 60)
    print("Phase 3: Computing metrics...")
    print("=" * 60)

    metrics, per_class_metrics = _compute_all_metrics(
        generated_images, all_labels, real_images, class_names,
        real_class_labels=real_labels,
        compute_fid_IS_per_class=args.compute_fid_IS_per_class,
        cache_dir=str(ckpt_dir),
    )
    metrics.update({
        'num_samples': float(num_samples),
        'num_steps': float(num_steps) if isinstance(num_steps, (int, float)) else 0.0,
        'cfg_scale': float(guidance_scale),
    })

    # ==============================================================
    # Phase 4: Save results
    # ==============================================================
    print("=" * 60)
    print("Phase 4: Saving results...")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_subdir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_subdir / f'{subfolder_name}.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Metrics saved to {csv_path}")

    png_path = eval_metrics.save_sample_grid(
        generated_images, str(output_dir), subfolder_name,
        labels=all_labels, class_names=class_names,
    )
    print(f"Sample grid saved to {png_path}")

    per_class_int = {m: {int(k): v for k, v in d.items()} for m, d in per_class_metrics.items()}
    per_class_csv = str(plot_subdir / f'{subfolder_name}_per_class.csv')
    eval_metrics.save_per_class_metrics(per_class_int, per_class_csv, class_names)
    print(f"Per-class metrics saved to {per_class_csv}")

    density_path = str(plot_subdir / f'{subfolder_name}_per_class_density.png')
    eval_metrics.save_density_plots(per_class_int, density_path, class_names)
    print(f"Density plot saved to {density_path}")

    if per_class_metrics.get('fid'):
        fid_grid_path = str(plot_subdir / f'{subfolder_name}_best_worst_fid.png')
        _save_best_worst_fid_grid(
            generated_images, all_labels,
            {int(k): v for k, v in per_class_metrics['fid'].items()},
            class_names, fid_grid_path,
        )
        print(f'Best/worst FID grid saved to {fid_grid_path}')

    print('=' * 60)
    print(f'FID:          {metrics["fid"]:.4f}')
    print(f'IS:           {metrics["is_mean"]:.4f} +/- {metrics["is_std"]:.4f}')
    print(f'CLIP Score:   {metrics["clip_score"]:.4f}')
    print(f'PickScore:    {metrics["pick_score"]:.4f}')
    print(f'CFG Scale:    {guidance_scale}')
    print(f'Steps:        {num_steps}')
    print(f'Samples:      {num_samples}')
    print(f'GPUs:         {num_gpus_total}')
    print(f'Output:       {csv_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
