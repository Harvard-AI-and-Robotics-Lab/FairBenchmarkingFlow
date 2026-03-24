"""Combine shards from parallel FLUX.1 [dev] generation and run metrics."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataloader import ImageNetDataset
from utils import eval as eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--shard_dir', type=str, required=True)
    parser.add_argument('--num_shards', type=int, required=True)
    parser.add_argument('--num_steps', type=int, required=True)
    parser.add_argument('--guidance_scale', type=float, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='flux1dev',
                        help='Name used for output CSV and PNG tag '
                             '(e.g. "flux1dev" or "natural_flux1dev")')
    parser.add_argument('--compute_fid_IS_per_class', action='store_true', default=False,
                        help='Compute per-class FID and IS (expensive — off by default)')
    args = parser.parse_args()

    # ---- Load and combine shards ----
    print("Combining shards...")
    all_images = []
    all_labels = []
    for i in range(args.num_shards):
        shard_path = os.path.join(args.shard_dir, f'shard_{i}.npz')
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        data = np.load(shard_path)
        all_images.append(data['images'])
        all_labels.append(data['labels'])
        print(f"  Shard {i}: {len(data['images'])} images")

    generated_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    num_samples = len(generated_images)
    print(f"Total: {generated_images.shape}")

    # Save combined checkpoint
    ckpt_path = os.path.join(args.shard_dir, 'generated_checkpoint.npz')
    np.savez_compressed(ckpt_path, images=generated_images, labels=all_labels)
    print(f"Combined checkpoint saved to {ckpt_path}")

    # ---- Load reference images ----
    print("=" * 60)
    print("Loading real ImageNet reference images...")
    print("=" * 60)

    ref_dataset = ImageNetDataset(
        data_path=args.data_path, split='val',
        image_size=args.image_size, return_uint8=True,
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
    print(f"Reference images: {real_images.shape}")

    # ---- Compute metrics ----
    print("=" * 60)
    print("Computing metrics...")
    print("=" * 60)

    print("  Computing FID...")
    fid_score = eval_metrics.compute_fid(real_images, generated_images)
    print(f"  FID: {fid_score:.4f}")

    print("  Computing Inception Score...")
    is_mean, is_std = eval_metrics.compute_inception_score(generated_images)
    print(f"  IS: {is_mean:.4f} +/- {is_std:.4f}")

    fid_per_class = None
    is_per_class = None
    if args.compute_fid_IS_per_class:
        print("  Computing per-class FID...")
        fid_per_class = eval_metrics.compute_fid_per_class(
            real_images, real_labels, generated_images, all_labels
        )
        print("  Computing per-class IS...")
        is_per_class = eval_metrics.compute_inception_score_per_class(
            generated_images, all_labels
        )

    print("  Computing CLIP Score...")
    clip_score, clip_per_class = eval_metrics.compute_clip_score(
        generated_images, all_labels, class_names, return_per_class=True)
    print(f"  CLIP Score: {clip_score:.4f}")

    prompts = [f"a photo of a {class_names[l]}" for l in all_labels]

    print("  Computing PickScore...")
    pick_score, pick_per_class = eval_metrics.compute_pick_score(
        generated_images, prompts, class_labels=all_labels, return_per_class=True)
    print(f"  PickScore: {pick_score:.4f}")

    # ---- Save results ----
    print("=" * 60)
    print("Saving results...")
    print("=" * 60)

    tag = f'{args.model_name}_steps{args.num_steps}'
    subfolder_name = f'{tag}_imagenet'
    plot_subdir = Path(args.shard_dir) / subfolder_name
    plot_subdir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_subdir / f'{subfolder_name}.csv'

    metrics = {
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std,
        'clip_score': clip_score,
        'pick_score': pick_score,
        'num_samples': float(num_samples),
        'num_steps': float(args.num_steps),
        'cfg_scale': float(args.guidance_scale),
    }

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Metrics saved to {csv_path}")

    # Save sample grid
    png_path = eval_metrics.save_sample_grid(
        generated_images, args.shard_dir, subfolder_name,
        labels=all_labels, class_names=class_names,
    )
    print(f"Sample grid saved to {png_path}")

    # Save per-class metrics CSV + density plot
    per_class_int = {
        'clip_score': {int(k): v for k, v in clip_per_class.items()},
        'pick_score': {int(k): v for k, v in pick_per_class.items()},
    }
    if fid_per_class is not None:
        per_class_int['fid'] = {int(k): v for k, v in fid_per_class.items()}
    if is_per_class is not None:
        per_class_int['inception_score'] = {int(k): v for k, v in is_per_class.items()}
    per_class_csv = str(plot_subdir / f'{subfolder_name}_per_class.csv')
    eval_metrics.save_per_class_metrics(per_class_int, per_class_csv, class_names)
    print(f"Per-class metrics saved to {per_class_csv}")

    density_path = str(plot_subdir / f'{subfolder_name}_per_class_density.png')
    eval_metrics.save_density_plots(per_class_int, density_path, class_names)
    print(f"Density plot saved to {density_path}")

    # ---- Summary ----
    print('=' * 60)
    print(f'FID:          {fid_score:.4f}')
    print(f'IS:           {is_mean:.4f} +/- {is_std:.4f}')
    print(f'CLIP Score:   {clip_score:.4f}')
    print(f'PickScore:    {pick_score:.4f}')
    print(f'Guidance:     {args.guidance_scale}')
    print(f'Steps:        {args.num_steps}')
    print(f'Samples:      {num_samples}')
    print(f'Output:       {csv_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
