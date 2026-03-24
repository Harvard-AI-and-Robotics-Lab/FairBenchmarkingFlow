"""Combine shards from parallel SD 3.5 generation and run metrics.

Called by run_sd35_eval.sh after all shard processes finish.
"""

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
    parser.add_argument('--seed', type=int, default=42)
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

    rng_np = np.random.RandomState(args.seed)
    real_indices = rng_np.choice(
        len(ref_dataset), size=min(num_samples, len(ref_dataset)), replace=False,
    )
    real_subset = torch.utils.data.Subset(ref_dataset, real_indices)
    real_loader = torch.utils.data.DataLoader(
        real_subset, batch_size=200, shuffle=False, num_workers=4, drop_last=False,
    )
    real_images = []
    for imgs, _ in real_loader:
        if isinstance(imgs, torch.Tensor):
            real_images.append(imgs.numpy())
        else:
            real_images.append(np.stack([np.array(img) for img in imgs]))
    real_images = np.concatenate(real_images, axis=0)
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

    print("  Computing CLIP Score...")
    clip_score = eval_metrics.compute_clip_score(
        generated_images, all_labels, class_names)
    print(f"  CLIP Score: {clip_score:.4f}")

    prompts = [f"a photo of a {class_names[l]}" for l in all_labels]

    print("  Computing PickScore...")
    pick_score = eval_metrics.compute_pick_score(generated_images, prompts)
    print(f"  PickScore: {pick_score:.4f}")

    print("  Computing ImageReward...")
    image_reward = eval_metrics.compute_image_reward(generated_images, prompts)
    print(f"  ImageReward: {image_reward:.4f}")

    # ---- Save results ----
    print("=" * 60)
    print("Saving results...")
    print("=" * 60)

    tag = f'sd35l_steps{args.num_steps}_cfg{args.guidance_scale}'
    csv_path = os.path.join(args.shard_dir, f'{tag}_imagenet.csv')

    metrics = {
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std,
        'clip_score': clip_score,
        'pick_score': pick_score,
        'image_reward': image_reward,
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

    sample_images = generated_images[:10]
    png_path = eval_metrics.save_sample_grid(
        sample_images, args.shard_dir, f'{tag}_imagenet')
    print(f"Sample grid saved to {png_path}")

    # ---- Summary ----
    print('=' * 60)
    print(f'FID:          {fid_score:.4f}')
    print(f'IS:           {is_mean:.4f} +/- {is_std:.4f}')
    print(f'CLIP Score:   {clip_score:.4f}')
    print(f'PickScore:    {pick_score:.4f}')
    print(f'ImageReward:  {image_reward:.4f}')
    print(f'Guidance:     {args.guidance_scale}')
    print(f'Steps:        {args.num_steps}')
    print(f'Samples:      {num_samples}')
    print(f'Output:       {csv_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
