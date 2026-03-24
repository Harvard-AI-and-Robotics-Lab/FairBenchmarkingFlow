"""Generate class-conditioned images using FLUX.1 [dev].

Generates num_samples images for a single ImageNet class using text prompts
("a photo of a {classname}") and saves each as a PNG.

Images are generated at 1024x1024 and resized to 256x256 (LANCZOS) by default.

--ckpt_path can be a HuggingFace model ID (e.g. "black-forest-labs/FLUX.1-dev")
or a path to a local directory containing the model weights.

Usage:
    python eval_generate.py \
        --ckpt_path black-forest-labs/FLUX.1-dev \
        --classname goldfish \
        --num_steps 50 \
        --cfg_scale 3.5 \
        --num_samples 4
"""

import gc
import json
import os
import sys
from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODEL_NAME = 'flux1'

# ======================================================================
# Class validation
# ======================================================================

def _load_imagenet_classes():
    json_path = (Path(__file__).resolve().parent.parent
                 / 'utils' / 'classifiers' / 'siglip_classifier' / 'imagenet_classes.json')
    with open(json_path) as f:
        raw = json.load(f)
    name_to_idx = {v[1]: int(k) for k, v in raw.items()}
    return name_to_idx


def _resolve_class(classname, name_to_idx):
    if classname in name_to_idx:
        return name_to_idx[classname]
    lower = {k.lower(): v for k, v in name_to_idx.items()}
    if classname.lower() in lower:
        return lower[classname.lower()]
    raise ValueError(
        f"Class '{classname}' not found in ImageNet classes.\n"
        f"Example valid names: tench, goldfish, great_white_shark"
    )


# ======================================================================
# Output path helpers
# ======================================================================

def _make_output_paths(output_path, repo_root, classname, cfg_scale, num_steps, num_samples):
    if output_path:
        if num_samples == 1:
            return [Path(output_path)]
        stem, ext = os.path.splitext(output_path)
        if not ext:
            ext = '.png'
        return [Path(f"{stem}_{i:03d}{ext}") for i in range(num_samples)]
    base = repo_root / 'outputs' / 'generations' / _MODEL_NAME
    tag = f"{classname}_{cfg_scale}_{num_steps}"
    return [base / f"{tag}_{i:03d}.png" for i in range(num_samples)]


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate images with FLUX.1 [dev]')
    parser.add_argument('--ckpt_path', required=True,
                        help='HuggingFace model ID or local path '
                             '(e.g. black-forest-labs/FLUX.1-dev)')
    parser.add_argument('--classname', required=True, help='ImageNet class name')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--cfg_scale', type=float, default=3.5, help='CFG guidance scale')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output_path', default=None, help='Override output path')
    parser.add_argument('--gen_size', type=int, default=1024,
                        help='Generation resolution (default: 1024)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Output image resolution after resize (default: 256)')
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    # Validate class name
    name_to_idx = _load_imagenet_classes()
    class_idx = _resolve_class(args.classname, name_to_idx)
    print(f"Class: '{args.classname}' → idx {class_idx}")

    # Build output paths
    out_paths = _make_output_paths(
        args.output_path, repo_root, args.classname,
        args.cfg_scale, args.num_steps, args.num_samples,
    )

    # ---- Load pipeline ----
    from diffusers import DiffusionPipeline

    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    print(f"Loading FLUX.1 [dev] pipeline from '{args.ckpt_path}'...")
    pipe = DiffusionPipeline.from_pretrained(args.ckpt_path, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Generate ----
    prompt = f"a photo of a {args.classname}"
    print(f"Prompt: '{prompt}'")
    print(f"Generating {args.num_samples} image(s) at {args.gen_size}×{args.gen_size} "
          f"→ resize to {args.image_size}×{args.image_size}, "
          f"steps={args.num_steps}, cfg={args.cfg_scale}...")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    images = []
    with torch.no_grad():
        for i in tqdm(range(args.num_samples), desc="Generating"):
            result = pipe(
                prompt=prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.cfg_scale,
                height=args.gen_size,
                width=args.gen_size,
                generator=generator,
            )
            img = result.images[0]
            if args.image_size != args.gen_size:
                img = img.resize((args.image_size, args.image_size), Image.LANCZOS)
            images.append(np.array(img))

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Save images ----
    for i, (img, out_path) in enumerate(zip(images, out_paths)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(str(out_path))
        print(f"Saved [{i+1}/{args.num_samples}]: {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
