"""Evaluate a SoFlow checkpoint on ImageNetV2 (matched-frequency).

Generates class-conditioned samples using the EMA model's solution operator,
decodes latents via the SD VAE, and computes FID, Inception Score, CLIP Score, and PickScore against real ImageNetV2 matched-frequency images.

SoFlow uses guidance distillation during training: the model learns to directly
output the CFG-guided velocity given real class labels, with no separate
unconditional forward pass needed at inference. NFE = num_steps exactly.

All ranks participate in generation via DDP; results are gathered to rank 0
which then computes metrics using src/utils/eval.py (DataParallel on all GPUs).

The YAML config and checkpoint .pt file are discovered automatically from the
weights directory (--checkpoint_path).  Pass --checkpoint_path to the model
weights directory (e.g. weights/SoFlow/B-2-cond); the script will:
  • find the *.yaml config inside that directory
  • find the latest ckpts/*.pt checkpoint inside that directory

Usage:
    torchrun --nproc_per_node=4 eval_imagenetv2.py \\
        --data_path=/path/to/ImageNetV2 \\
        --checkpoint_path=weights/SoFlow/B-2-cond \\
        --num_steps=1 \\
        --cfg=7.0 \\
        --model_name=soflow
"""
import argparse
import csv
import gc
import json
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
import yaml
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from models import Soflow

# Add src/ to import path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataloader import ImageNetV2Dataset
from utils import eval as eval_metrics


# ======================================================================
# Metrics helpers (mirrors meanflow / imeanflow pattern)
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
    """Compute FID, IS, CLIP Score, and PickScore with optional caching.

    Returns (global_dict, per_class_dict). When *cache_dir* is provided each
    metric is saved to a JSON file as soon as it finishes.  On a subsequent
    call with the same *cache_dir*, already-cached metrics are loaded instead
    of recomputed, so a crash mid-way through doesn't lose earlier results.
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
        if 'fid_per_class' not in cached or not cached['fid_per_class']:
            print('Computing per-class FID...')
            fid_pc = eval_metrics.compute_fid_per_class(
                real_images, real_class_labels, generated_images, all_labels
            )
            per_class['fid'] = {str(k): v for k, v in fid_pc.items()}
            _cache(results, per_class)
        else:
            print('FID per-class: cached')
            per_class['fid'] = {str(k): v for k, v in cached['fid_per_class'].items()}

        if 'is_per_class' not in cached or not cached['is_per_class']:
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
# Main evaluation
# ======================================================================

def evaluate(args, model_config):
    """DDP generation → gather to rank 0 → metrics → save.

    Phases:
      1. All ranks: load EMA model + VAE, generate latents, decode inline,
         convert to uint8 NHWC, gather to rank 0.
      2. Rank 0: load real ImageNetV2 matched-frequency images via ImageNetV2Dataset.
      3. Rank 0: compute FID / IS / CLIP / PickScore.
      4. Rank 0: save CSV + sample grid PNG, clean up temp files.
    """
    # ---- DDP setup ----
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.manual_seed(args.seed + rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(
        backend='nccl', init_method='env://',
        rank=rank, world_size=world_size, device_id=device,
        timeout=timedelta(hours=2),
    )
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_steps = args.num_steps

    # ---- Load val labels on all ranks (lightweight — reads file paths only) ----
    _meta = ImageNetV2Dataset(data_path=args.data_path, image_size=256, return_uint8=True)
    val_labels_all = np.array([s[1] for s in _meta.samples])  # (~10000,) ordered by dataset
    num_samples = len(val_labels_all)
    del _meta

    # ---- Output paths (created by rank 0) ----
    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = os.path.join(repo_root, 'outputs', 'soflow', f'steps_{num_steps}')
    subfolder_name = f'{args.model_name}_steps_{num_steps}_imagenetv2'
    plot_subdir = os.path.join(output_dir, subfolder_name)
    ckpt_dir = os.path.join(plot_subdir, '_checkpoints')
    generated_path = os.path.join(ckpt_dir, 'generated.npz')

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()

    skip_gen = args.resume and os.path.exists(generated_path)

    # ==================================================================
    # Phase 1: DDP generation + inline VAE decode → gather to rank 0
    # ==================================================================
    if skip_gen:
        if rank == 0:
            print(f'Phase 1: SKIPPED (resuming from {generated_path})')
    else:
        # ---- Model construction params from YAML ----
        data_channels = 4
        data_size = model_config.get('data_size', 32)
        num_classes = 1000
        noising_type = model_config.get('noising_type', 'Linear')
        coefficient_type = model_config.get('coefficient_type', 'Euler')
        model_type = model_config.get('model_type', 'DiT-B-4')
        cfg_drop_rate = model_config.get('cfg_drop_rate', 0.1)

        # ---- Load EMA model ----
        ema_model = Soflow(
            data_channels=data_channels,
            data_size=data_size,
            num_classes=num_classes,
            cfg_drop_rate=cfg_drop_rate,
            noising_type=noising_type,
            coefficient_type=coefficient_type,
            model_type=model_type,
        ).to(device)

        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        if not args.no_ema:
            ema_model.load_state_dict(ckpt['ema_model'])
            if rank == 0:
                print('Using EMA params.')
        else:
            ema_model.load_state_dict(ckpt['model'])
            if rank == 0:
                print('Using raw model params.')
        del ckpt

        ema_model.requires_grad_(False)
        ema_model.eval()

        # ---- VAE ----
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device)
        vae.requires_grad_(False)
        vae.eval()

        # ---- Time schedule (matches inference.py) ----
        eval_time = [1.0 - i / num_steps for i in range(num_steps + 1)]

        # Distribute val labels across ranks — each rank gets its contiguous slice.
        n_per_rank = (num_samples + world_size - 1) // world_size
        rank_start = rank * n_per_rank
        rank_end = min(rank_start + n_per_rank, num_samples)
        rank_labels = val_labels_all[rank_start:rank_end]   # exact slice, no random
        generate_per_rank = len(rank_labels)
        batch_size = args.batch_size

        all_images_local = []   # list of uint8 NHWC CPU tensors
        all_labels_local = []   # list of int64 CPU tensors

        with torch.inference_mode():
            for start in tqdm(
                range(0, generate_per_rank, batch_size),
                desc=f'Generate [rank {rank}]',
                disable=(rank != 0),
            ):
                end = min(generate_per_rank, start + batch_size)
                size = end - start

                t_ones = torch.ones((size,), device=device)
                labels = torch.tensor(rank_labels[start:end], dtype=torch.long, device=device)
                x_t = torch.randn(
                    (size, data_channels, data_size, data_size), device=device,
                )

                # Multi-step solution flow schedule
                # Guidance is distilled into the model during training — single
                # forward pass per step, NFE = num_steps exactly.
                for i in range(len(eval_time) - 2):
                    t_i = eval_time[i] * t_ones
                    x_t = ema_model.solution_operator(x_t, t_i, torch.zeros_like(t_i), labels)
                    # Re-noise for next step
                    next_t = eval_time[i + 1] * t_ones
                    alpha_next = ema_model.scheduler.alpha(next_t).view(-1, 1, 1, 1)
                    beta_next = ema_model.scheduler.beta(next_t).view(-1, 1, 1, 1)
                    x_t = alpha_next * x_t + beta_next * torch.randn_like(x_t)

                # Final step
                t_last = eval_time[-2] * t_ones
                t_end_val = eval_time[-1] * t_ones  # = 0.0
                x_t = ema_model.solution_operator(x_t, t_last, t_end_val, labels)

                # Decode latents with VAE
                x_decoded = vae.decode(x_t / 0.18215).sample  # NCHW float

                # Convert to uint8 NHWC (matches eval.py expected format)
                samples = (x_decoded * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
                samples = samples.permute(0, 2, 3, 1).contiguous()  # NHWC

                all_images_local.append(samples.cpu())
                all_labels_local.append(labels.cpu())

        local_images = torch.cat(all_images_local, dim=0)   # (N_local, H, W, C)
        local_labels = torch.cat(all_labels_local, dim=0)   # (N_local,)
        del ema_model, vae, all_images_local, all_labels_local
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Gather to rank 0 ----
        # All ranks must have the same tensor size for dist.gather, so pad
        # to the maximum size across ranks.
        max_local = torch.tensor(local_images.size(0), dtype=torch.long, device=device)
        dist.all_reduce(max_local, op=dist.ReduceOp.MAX)
        max_local = max_local.item()

        if local_images.size(0) < max_local:
            pad_n = max_local - local_images.size(0)
            local_images = torch.cat([
                local_images,
                torch.zeros(pad_n, *local_images.shape[1:], dtype=local_images.dtype),
            ], dim=0)
            local_labels = torch.cat([
                local_labels,
                torch.zeros(pad_n, dtype=local_labels.dtype),
            ], dim=0)

        local_images_dev = local_images.to(device)
        local_labels_dev = local_labels.to(device)

        if rank == 0:
            gathered_images = [torch.zeros_like(local_images_dev) for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(local_labels_dev) for _ in range(world_size)]
        else:
            gathered_images = None
            gathered_labels = None

        dist.gather(local_images_dev, gather_list=gathered_images, dst=0)
        dist.gather(local_labels_dev, gather_list=gathered_labels, dst=0)

        if rank == 0:
            all_images = torch.cat(gathered_images, dim=0)[:num_samples]
            generated_images_np = all_images.cpu().numpy()
            # Labels are the v2 dataset labels in order (rank 0 slice first, then rank 1, …)
            np.savez(generated_path, images=generated_images_np, labels=val_labels_all)
            print(f'Phase 1 done: {generated_images_np.shape[0]} images saved to {generated_path}')

    # ---- Non-rank-0 processes exit; rank 0 continues with metrics ----
    dist.destroy_process_group()
    if rank != 0:
        return

    # ==================================================================
    # Rank 0 only from here
    # ==================================================================

    # Always reload from npz so gather tensors are freed before metrics
    data = np.load(generated_path)
    generated_images = data['images']    # uint8 NHWC numpy
    all_labels = data['labels'].astype(int)
    print(f'Loaded {generated_images.shape[0]} generated images.')
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Phase 2: Load ALL reference images (ImageNetV2 matched-frequency)
    # ==================================================================
    print('Phase 2: Loading all real ImageNetV2 matched-frequency images for reference...')
    ref_dataset = ImageNetV2Dataset(
        data_path=args.data_path,
        image_size=256,
        return_uint8=True,
    )
    class_names = ref_dataset.class_names
    real_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=200, shuffle=False, num_workers=4, drop_last=False,
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
    print(f'Reference images: {real_images.shape}')

    # ==================================================================
    # Phase 3: Compute metrics (with per-metric caching on resume)
    # ==================================================================
    print('Phase 3: Computing metrics...')
    metrics, per_class_metrics = _compute_all_metrics(
        generated_images, all_labels, real_images, class_names,
        real_class_labels=real_labels,
        compute_fid_IS_per_class=args.compute_fid_IS_per_class,
        cache_dir=ckpt_dir if args.resume else None,
    )
    metrics.update({
        'num_samples': float(num_samples),
        'num_steps': float(num_steps),
        'cfg_scale': float(args.cfg_scale),
    })

    # ==================================================================
    # Phase 4: Save results and clean up
    # ==================================================================
    os.makedirs(plot_subdir, exist_ok=True)

    csv_path = os.path.join(plot_subdir, f'{subfolder_name}.csv')
    _save_metrics(metrics, csv_path)
    print(f'Metrics saved to {csv_path}')

    # Save sample grid
    png_path = eval_metrics.save_sample_grid(
        generated_images, output_dir, subfolder_name,
        labels=all_labels, class_names=class_names,
    )
    print(f'Sample grid saved to {png_path}')

    # Save per-class metrics CSV + density plot (in same subfolder as grid)
    per_class_int = {m: {int(k): v for k, v in d.items()} for m, d in per_class_metrics.items()}
    per_class_csv = os.path.join(plot_subdir, f'{subfolder_name}_per_class.csv')
    eval_metrics.save_per_class_metrics(per_class_int, per_class_csv, class_names)
    print(f'Per-class metrics saved to {per_class_csv}')

    density_path = os.path.join(plot_subdir, f'{subfolder_name}_per_class_density.png')
    eval_metrics.save_density_plots(per_class_int, density_path, class_names)
    print(f'Density plot saved to {density_path}')

    print('=' * 60)
    print(f'FID:          {metrics["fid"]:.4f}')
    print(f'IS:           {metrics["is_mean"]:.4f} +/- {metrics["is_std"]:.4f}')
    print(f'CLIP Score:   {metrics["clip_score"]:.4f}')
    print(f'PickScore:    {metrics["pick_score"]:.4f}')
    print(f'CFG Scale:    {args.cfg_scale}')
    print(f'Steps:        {num_steps}')
    print(f'Output:       {csv_path}')
    print('=' * 60)


def _resolve_weights_dir(checkpoint_path):
    """Given a weights directory, locate the YAML config and .pt checkpoint.

    Expects the layout produced by the HuggingFace SoFlow repo:
        <checkpoint_path>/
            *.yaml          ← training config
            ckpts/*.pt      ← one or more checkpoints (latest by mtime wins)
    """
    weights_dir = Path(checkpoint_path)
    if not weights_dir.is_dir():
        raise ValueError(f'--checkpoint_path must be a directory, got: {checkpoint_path}')

    # Find YAML config
    yaml_files = list(weights_dir.glob('*.yaml'))
    if not yaml_files:
        raise FileNotFoundError(f'No *.yaml config found in {weights_dir}')
    config_path = yaml_files[0]
    if len(yaml_files) > 1:
        print(f'[warn] Multiple YAML files found, using {config_path}')

    # Find .pt checkpoint (latest by modification time)
    pt_files = list((weights_dir / 'ckpts').glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f'No ckpts/*.pt found under {weights_dir}')
    pt_path = max(pt_files, key=lambda p: p.stat().st_mtime)

    return config_path, pt_path


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a SoFlow checkpoint on ImageNetV2 matched-frequency (FID, IS, CLIP, PickScore)'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNetV2 root directory (contains '
                             'imagenetv2-matched-frequency-format-val/)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the SoFlow weights directory (e.g. weights/SoFlow/B-2-cond). '
                             'The YAML config and ckpts/*.pt are discovered automatically.')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Number of sampling steps = NFE (default: 1)')
    parser.add_argument('--cfg', type=float, default=1.0,
                        help='For record-keeping only — SoFlow bakes guidance into the model '
                             'during training so no external CFG is applied at inference. '
                             'The actual guidance scale is read from the training config.')
    parser.add_argument('--model_name', type=str, default='soflow',
                        help='Name used for output file naming (default: soflow)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Per-device batch size for sampling (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no_ema', action='store_true',
                        help='Use raw model params instead of EMA params')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from cached outputs (skip completed phases)')
    parser.add_argument('--compute_fid_IS_per_class', action='store_true', default=False,
                        help='Compute per-class FID and IS (expensive — off by default)')
    args = parser.parse_args()

    # Resolve checkpoint_path relative to the script's directory when given as a relative path
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = script_dir / checkpoint_path
    args.checkpoint_path = str(checkpoint_path)

    # Discover config + checkpoint from weights directory
    config_path, pt_path = _resolve_weights_dir(args.checkpoint_path)
    print(f'Config:     {config_path}')
    print(f'Checkpoint: {pt_path}')

    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Resolve actual checkpoint .pt path
    args.checkpoint_path = str(pt_path)
    # Alias so evaluate() can use args.cfg_scale unchanged
    args.cfg_scale = args.cfg

    ref = ImageNetV2Dataset(data_path=args.data_path, return_uint8=True)
    num_v2 = len(ref)
    del ref
    print(f'Evaluating with all {num_v2} ImageNetV2 matched-frequency samples')

    evaluate(args, model_config)


if __name__ == '__main__':
    main()
