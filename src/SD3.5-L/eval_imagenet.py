"""Evaluate SD 3.5 Large on ImageNet.

Generates class-conditioned samples using text prompts ("a photo of a {class}"),
decodes via the built-in VAE, resizes to 256x256, and computes FID, Inception
Score, CLIP Score, PickScore, and ImageReward against real ImageNet images.

Uses the HuggingFace diffusers StableDiffusion3Pipeline for inference.
SD 3.5 Large is a text-to-image model (not class-conditioned like RAE/MeanFlow),
so we condition on ImageNet class names via text prompts.

Follows the same phase structure as RAE/MeanFlow eval scripts:
    Phase 1: Generate images (SD3.5 pipeline)
    Phase 2: Load real ImageNet references (shared dataloader)
    Phase 3: Compute metrics (shared eval)
    Phase 4: Save results

Supports multi-GPU sharding:
    # GPU 0 of 4
    CUDA_VISIBLE_DEVICES=0 python eval_imagenet.py \
        --data_path /path/to/imagenet --shard_id 0 --num_shards 4 \
        --shard_dir outputs/shards --num_steps 25 --guidance_scale 3.5

Usage:
    # 5-sample sanity check
    python eval_imagenet.py \
        --data_path /path/to/imagenet \
        --num_samples 5 --num_steps 28

    # 50k FID-50k benchmark
    python eval_imagenet.py \
        --data_path /path/to/imagenet \
        --test --num_steps 28
"""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tqdm import tqdm

# Add src/ to import path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataloader import ImageNetDataset
from utils import eval as eval_metrics


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
        # All unique classes expected in the generated set (used for both checks below)
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


def generate_images(num_samples, num_steps, guidance_scale, seed, batch_size,
                    image_size, class_names, val_labels, precision,
                    ckpt_every, output_dir, partial_tag=""):
    """Generate images using SD 3.5 Large via diffusers pipeline.

    Args:
        num_samples: Total number of images to generate.
        num_steps: Number of denoising steps.
        guidance_scale: CFG guidance scale.
        seed: Random seed.
        batch_size: Batch size for generation.
        image_size: Target image resolution for metrics (images are generated
                    at 1024x1024 then resized).
        class_names: List of ImageNet class name strings.
        val_labels: numpy array of integer class labels ordered by the val dataset.
        precision: 'bf16' or 'fp32'.
        ckpt_every: Save checkpoint every N batches.
        output_dir: Directory for checkpoints.

    Returns:
        generated_images: (N, H, W, 3) uint8 numpy array at image_size.
        all_labels: (N,) int numpy array of class indices.
    """
    from diffusers import StableDiffusion3Pipeline

    # ---- Load pipeline ----
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float32
    print("Loading SD 3.5 Large pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=dtype,
    )
    pipe = pipe.to("cuda")

    # Disable safety checker if present (for benchmarking)
    if hasattr(pipe, 'safety_checker'):
        pipe.safety_checker = None

    print(f"Generating {num_samples} images at 1024x1024 -> resize to {image_size}x{image_size}")
    print(f"  Steps: {num_steps}, Guidance: {guidance_scale}, Batch: {batch_size}")

    # ---- Check for partial checkpoint ----
    os.makedirs(output_dir, exist_ok=True)
    partial_path = os.path.join(output_dir, f'_partial_checkpoint{partial_tag}.npz')

    all_images = []
    all_labels = []
    num_generated = 0

    if os.path.exists(partial_path):
        print(f"Resuming from partial checkpoint...")
        data = np.load(partial_path)
        all_images = list(data['images'])
        all_labels = list(data['labels'])
        num_generated = int(data['num_generated'])
        print(f"  Resumed: {num_generated}/{num_samples} images")

    # ---- Generation loop ----
    generator = torch.Generator(device="cuda").manual_seed(seed + num_generated)
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(total=num_samples, initial=num_generated, desc="Generating")
        while num_generated < num_samples:
            current_batch = min(batch_size, num_samples - num_generated)

            # Val class labels (ordered by the val dataset)
            labels = val_labels[num_generated:num_generated + current_batch]
            prompts = [f"a photo of a {class_names[l]}" for l in labels]

            # Generate at 1024x1024 (native resolution)
            result = pipe(
                prompt=prompts,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
                generator=generator,
            )

            # Resize to target size and convert to uint8 numpy
            batch_images = []
            for img in result.images:
                img_resized = img.resize((image_size, image_size), Image.LANCZOS)
                batch_images.append(np.array(img_resized))

            all_images.append(np.stack(batch_images))
            all_labels.append(labels)
            num_generated += current_batch
            batch_count += 1
            pbar.update(current_batch)

            # Periodic checkpoint
            if ckpt_every > 0 and batch_count % ckpt_every == 0:
                _imgs = np.concatenate(all_images, axis=0)
                _lbls = np.concatenate(all_labels, axis=0)
                np.savez(partial_path, images=_imgs, labels=_lbls,
                         num_generated=np.array(num_generated))
                del _imgs, _lbls
                print(f"\n  Checkpoint saved ({num_generated}/{num_samples})")

        pbar.close()

    # ---- Concatenate and clean up ----
    generated_images = np.concatenate(all_images, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    # Free pipeline memory
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return generated_images, all_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate SD 3.5 Large on ImageNet")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet root')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Override number of samples (for sanity checks)')
    parser.add_argument('--test', action='store_true',
                        help='Run with 50,000 samples (standard FID-50k)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation (default 4, SD3.5 is memory heavy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--precision', type=str, default='bf16',
                        choices=['fp32', 'bf16'],
                        help='Inference precision (default bf16)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Target image resolution for metrics')
    parser.add_argument('--num_steps', type=int, default=28,
                        help='Number of denoising steps (default 28)')
    parser.add_argument('--guidance_scale', type=float, default=3.5,
                        help='CFG guidance scale (default 3.5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to generated_checkpoint.npz to skip generation')
    parser.add_argument('--ckpt_every', type=int, default=50,
                        help='Save partial checkpoint every N batches (0 to disable)')
    parser.add_argument('--model_name', type=str, default='sd35l',
                        help='Name used for output CSV and PNG tag '
                             '(e.g. "sd35l" or "natural_sd35l")')
    parser.add_argument('--compute_fid_IS_per_class', action='store_true', default=False,
                        help='Compute per-class FID and IS (expensive — off by default)')
    # Multi-GPU sharding
    parser.add_argument('--shard_id', type=int, default=None,
                        help='Shard index (0-based). If set, only generates '
                             'this shard and saves shard_{id}.npz (no metrics).')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of shards for parallel generation.')
    parser.add_argument('--shard_dir', type=str, default=None,
                        help='Directory to save shard .npz files.')
    args = parser.parse_args()

    # ---- Determine number of samples ----
    if args.num_samples is not None:
        num_samples = args.num_samples
    elif args.test:
        num_samples = 50000
    else:
        ds = ImageNetDataset(data_path=args.data_path, split='val',
                             image_size=args.image_size, return_uint8=True)
        num_samples = len(ds)
        del ds

    # ==================================================================
    # SHARD MODE: generate only this shard's portion and exit
    # ==================================================================
    if args.shard_id is not None:
        assert args.shard_dir is not None, "--shard_dir required when using --shard_id"
        os.makedirs(args.shard_dir, exist_ok=True)

        # Split samples across shards
        all_indices = np.arange(num_samples)
        shard_splits = np.array_split(all_indices, args.num_shards)
        my_indices = shard_splits[args.shard_id]
        shard_num_samples = len(my_indices)
        shard_start = int(my_indices[0])

        print(f"SHARD MODE: shard {args.shard_id}/{args.num_shards}")
        print(f"  Generating samples {shard_start} to {shard_start + shard_num_samples - 1} "
              f"({shard_num_samples} images)")

        # Load class names and val labels
        ref_ds = ImageNetDataset(data_path=args.data_path, split='val',
                                 image_size=args.image_size, return_uint8=True)
        class_names = ref_ds.class_names
        val_labels_all = np.array([s[1] for s in ref_ds.samples], dtype=np.int32)
        del ref_ds
        if np.any(val_labels_all < 0):
            raise ValueError(
                "ImageNet val directory appears to use a flat structure (no synset "
                "subdirectories), so class labels are -1. Python's negative indexing "
                "would silently use the last class for all samples. Please organize "
                "val/ into synset subdirectories (e.g., val/n01440764/image.JPEG)."
            )

        # Get this shard's labels
        shard_labels = val_labels_all[my_indices]

        # Use a different seed offset per shard for diversity
        shard_seed = args.seed + args.shard_id * 10000

        generated_images, gen_labels = generate_images(
            num_samples=shard_num_samples,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=shard_seed,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_names=class_names,
            val_labels=shard_labels,
            precision=args.precision,
            ckpt_every=args.ckpt_every,
            output_dir=args.shard_dir,
            partial_tag=f"_shard{args.shard_id}",
        )

        # Save shard
        shard_path = os.path.join(args.shard_dir, f'shard_{args.shard_id}.npz')
        np.savez_compressed(shard_path, images=generated_images, labels=gen_labels)
        print(f"Shard {args.shard_id} saved to {shard_path} ({len(generated_images)} images)")
        return  # Exit — combine_sd35_shards.py handles metrics

    # ==================================================================
    # SINGLE-GPU MODE: original behavior
    # ==================================================================
    print(f"SD 3.5 Large ImageNet Evaluation")
    print(f"  Samples: {num_samples}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Precision: {args.precision}")
    print(f"  Image size: {args.image_size}")
    print("=" * 60)

    # ---- Output directory ----
    script_dir = Path(__file__).resolve().parent
    folder_name = f'steps_{args.num_steps}'
    output_dir = script_dir / 'outputs' / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.model_name}_steps{args.num_steps}'
    subfolder_name = f'{tag}_imagenet'
    plot_subdir = output_dir / subfolder_name
    ckpt_dir = plot_subdir / '_checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Phase 1: Generate images (or resume from checkpoint)
    # ==================================================================
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = np.load(args.resume)
        generated_images = ckpt['images'][:num_samples]
        all_labels = ckpt['labels'][:num_samples]
        del ckpt
        print(f"Loaded {generated_images.shape[0]} images from checkpoint.")
    else:
        print("=" * 60)
        print("Phase 1: Generating images with SD 3.5 Large...")
        print("=" * 60)

        # Load class names and val labels from the dataset
        ref_ds = ImageNetDataset(data_path=args.data_path, split='val',
                                 image_size=args.image_size, return_uint8=True)
        class_names = ref_ds.class_names
        val_labels_all = np.array([s[1] for s in ref_ds.samples], dtype=np.int32)
        del ref_ds
        if np.any(val_labels_all < 0):
            raise ValueError(
                "ImageNet val directory appears to use a flat structure (no synset "
                "subdirectories), so class labels are -1. Python's negative indexing "
                "would silently use the last class for all samples. Please organize "
                "val/ into synset subdirectories (e.g., val/n01440764/image.JPEG)."
            )

        generated_images, all_labels = generate_images(
            num_samples=num_samples,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_names=class_names,
            val_labels=val_labels_all[:num_samples],
            precision=args.precision,
            ckpt_every=args.ckpt_every,
            output_dir=str(output_dir),
        )

        print(f"Phase 1 done: {generated_images.shape}")

        # Save checkpoint
        ckpt_path = output_dir / 'generated_checkpoint.npz'
        print(f"Saving checkpoint to {ckpt_path}...")
        np.savez_compressed(str(ckpt_path),
                            images=generated_images, labels=all_labels)
        print("Checkpoint saved.")

    # ==================================================================
    # Phase 2: Load real ImageNet reference images
    # ==================================================================
    print("=" * 60)
    print("Phase 2: Loading real ImageNet reference images...")
    print("=" * 60)

    ref_dataset = ImageNetDataset(
        data_path=args.data_path,
        split='val',
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

    # ==================================================================
    # Phase 3: Compute metrics (with per-metric caching on resume)
    # ==================================================================
    print("=" * 60)
    print("Phase 3: Computing metrics...")
    print("=" * 60)

    metrics, per_class_metrics = _compute_all_metrics(
        generated_images, all_labels, real_images, class_names,
        real_class_labels=real_labels,
        compute_fid_IS_per_class=args.compute_fid_IS_per_class,
        cache_dir=str(ckpt_dir),          # always cache; per-class metrics are expensive
    )
    metrics.update({
        'num_samples': float(num_samples),
        'num_steps': float(args.num_steps),
        'cfg_scale': float(args.guidance_scale),
    })

    # ==================================================================
    # Phase 4: Save results
    # ==================================================================
    print("=" * 60)
    print("Phase 4: Saving results...")
    print("=" * 60)

    csv_path = plot_subdir / f'{subfolder_name}.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Metrics saved to {csv_path}")

    # Save sample grid
    png_path = eval_metrics.save_sample_grid(
        generated_images, str(output_dir), subfolder_name,
        labels=all_labels, class_names=class_names,
    )
    print(f"Sample grid saved to {png_path}")

    # Save per-class metrics CSV + density plot
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

    # ---- Summary ----
    print('=' * 60)
    print(f'FID:          {metrics["fid"]:.4f}')
    print(f'IS:           {metrics["is_mean"]:.4f} +/- {metrics["is_std"]:.4f}')
    print(f'CLIP Score:   {metrics["clip_score"]:.4f}')
    print(f'PickScore:    {metrics["pick_score"]:.4f}')
    print(f'Guidance:     {args.guidance_scale}')
    print(f'Steps:        {args.num_steps}')
    print(f'Samples:      {num_samples}')
    print(f'Output:       {csv_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()