"""Generate a shard of SD 3.5 Large images (single GPU).

Called by run_sd35_eval.sh — one instance per GPU. Each shard generates
num_samples images and saves to a checkpoint file. The combine script
merges all shards and runs metrics.
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataloader import ImageNetDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--num_steps', type=int, default=28)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--shard_id', type=int, required=True)
    parser.add_argument('--num_shards', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt_every', type=int, default=25)
    args = parser.parse_args()

    print(f"[Shard {args.shard_id}] Starting: {args.num_samples} images, "
          f"steps={args.num_steps}, cfg={args.guidance_scale}")

    # ---- Check for existing completed shard ----
    shard_path = os.path.join(args.output_dir, f'shard_{args.shard_id}.npz')
    if os.path.exists(shard_path):
        data = np.load(shard_path)
        if len(data['images']) >= args.num_samples:
            print(f"[Shard {args.shard_id}] Already complete, skipping.")
            return

    # ---- Load class names ----
    ref_ds = ImageNetDataset(data_path=args.data_path, split='val',
                             image_size=args.image_size, return_uint8=True)
    class_names = ref_ds.class_names
    num_classes = len(class_names)
    del ref_ds

    # ---- Load pipeline ----
    from diffusers import StableDiffusion3Pipeline

    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    print(f"[Shard {args.shard_id}] Loading SD 3.5 Large pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=dtype,
    )
    pipe = pipe.to("cuda")

    if hasattr(pipe, 'safety_checker'):
        pipe.safety_checker = None

    # ---- Check for partial checkpoint ----
    os.makedirs(args.output_dir, exist_ok=True)
    partial_path = os.path.join(args.output_dir, f'shard_{args.shard_id}_partial.npz')

    all_images = []
    all_labels = []
    num_generated = 0

    if os.path.exists(partial_path):
        data = np.load(partial_path)
        all_images = [data['images']]
        all_labels = [data['labels']]
        num_generated = int(data['num_generated'])
        print(f"[Shard {args.shard_id}] Resumed from {num_generated}/{args.num_samples}")

    # ---- Generate ----
    rng = np.random.RandomState(args.seed)
    if num_generated > 0:
        _ = rng.randint(0, num_classes, size=num_generated)

    generator = torch.Generator(device="cuda").manual_seed(args.seed + num_generated)
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, initial=num_generated,
                    desc=f'Shard {args.shard_id}')
        while num_generated < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - num_generated)

            labels = rng.randint(0, num_classes, size=current_batch)
            prompts = [f"a photo of a {class_names[l]}" for l in labels]

            result = pipe(
                prompt=prompts,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                height=1024,
                width=1024,
                generator=generator,
            )

            batch_images = []
            for img in result.images:
                img_resized = img.resize((args.image_size, args.image_size), Image.LANCZOS)
                batch_images.append(np.array(img_resized))

            all_images.append(np.stack(batch_images))
            all_labels.append(labels)
            num_generated += current_batch
            batch_count += 1
            pbar.update(current_batch)

            # Periodic checkpoint
            if args.ckpt_every > 0 and batch_count % args.ckpt_every == 0:
                _imgs = np.concatenate(all_images, axis=0)
                _lbls = np.concatenate(all_labels, axis=0)
                np.savez(partial_path, images=_imgs, labels=_lbls,
                         num_generated=np.array(num_generated))
                del _imgs, _lbls
                print(f"\n[Shard {args.shard_id}] Checkpoint: {num_generated}/{args.num_samples}")

        pbar.close()

    # ---- Save final shard ----
    generated_images = np.concatenate(all_images, axis=0)[:args.num_samples]
    all_labels_arr = np.concatenate(all_labels, axis=0)[:args.num_samples]

    np.savez_compressed(shard_path, images=generated_images, labels=all_labels_arr)
    print(f"[Shard {args.shard_id}] Saved {len(generated_images)} images to {shard_path}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
