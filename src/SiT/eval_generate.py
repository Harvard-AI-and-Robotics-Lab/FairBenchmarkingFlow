"""Generate class-conditioned images from a SiT checkpoint.

Generates num_samples images for a single ImageNet class using CFG (ODE mode)
and saves each as a PNG.

Usage:
    python eval_generate.py \
        --ckpt_path weights/SiT-XL-2-256x256.pt \
        --classname goldfish \
        --num_steps 250 \
        --cfg_scale 4.0 \
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
from diffusers.models import AutoencoderKL

from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_transport_args
from transport import create_transport, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODEL_NAME = 'sit'

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
# Model loading (mirrors eval_imagenet.py _load_sit_model)
# ======================================================================

def _load_sit_model(ckpt_path, model_name, image_size, num_classes, device):
    latent_size = image_size // 8
    state_dict = find_model(ckpt_path)

    load_errors = []
    for learn_sigma in (False, True):
        model = SiT_models[model_name](
            input_size=latent_size,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        ).to(device)
        try:
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[info] Loaded model with learn_sigma={learn_sigma}")
            return model
        except RuntimeError as err:
            load_errors.append(str(err))

    raise RuntimeError(
        "Failed to load checkpoint into SiT model with both learn_sigma variants.\n"
        + "\n---\n".join(load_errors)
    )


def _build_sample_fn(args):
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    sampler = Sampler(transport)
    return sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.num_sampling_steps,
        atol=args.atol,
        rtol=args.rtol,
        reverse=args.reverse,
    )


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate images with SiT')
    parser.add_argument('--ckpt_path', required=True, help='Path to SiT checkpoint .pt file')
    parser.add_argument('--classname', required=True, help='ImageNet class name')
    parser.add_argument('--num_steps', type=int, default=250, help='Number of ODE sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=4.0, help='CFG guidance scale')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output_path', default=None, help='Override output path')
    parser.add_argument('--model', type=str, default='SiT-XL/2', choices=list(SiT_models.keys()),
                        help='SiT model variant')
    parser.add_argument('--vae', type=str, default='mse', choices=['ema', 'mse'])
    parser.add_argument('--image_size', type=int, default=256, choices=[256, 512])
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Transport and ODE args (same defaults as eval_imagenet.py)
    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_args()

    # num_sampling_steps used by _build_sample_fn
    args.num_sampling_steps = args.num_steps

    repo_root = Path(__file__).resolve().parent.parent.parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate class name
    name_to_idx = _load_imagenet_classes()
    class_idx = _resolve_class(args.classname, name_to_idx)
    print(f"Class: '{args.classname}' → idx {class_idx}")

    # Build output paths
    out_paths = _make_output_paths(
        args.output_path, repo_root, args.classname,
        args.cfg_scale, args.num_steps, args.num_samples,
    )

    # ---- Load model + VAE ----
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading SiT model ({args.model}) from {args.ckpt_path}...")
    model = _load_sit_model(args.ckpt_path, args.model, args.image_size, args.num_classes, device)

    vae_id = f"stabilityai/sd-vae-ft-{args.vae}"
    print(f"Loading VAE ({vae_id})...")
    vae = AutoencoderKL.from_pretrained(vae_id).to(device)
    vae.eval()

    sample_fn = _build_sample_fn(args)

    # ---- Generate ----
    latent_size = args.image_size // 8
    y = torch.full((args.num_samples,), class_idx, dtype=torch.long, device=device)

    print(f"Generating {args.num_samples} image(s) for class '{args.classname}' "
          f"(idx={class_idx}), steps={args.num_steps}, cfg={args.cfg_scale}...")

    with torch.inference_mode():
        z = torch.randn(args.num_samples, model.in_channels, latent_size, latent_size, device=device)

        using_cfg = args.cfg_scale > 1.0
        if using_cfg:
            z_in = torch.cat([z, z], dim=0)
            y_null = torch.full((args.num_samples,), args.num_classes, device=device, dtype=y.dtype)
            y_in = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y_in, cfg_scale=float(args.cfg_scale))
            model_fn = model.forward_with_cfg
        else:
            z_in = z
            y_in = y
            model_kwargs = dict(y=y_in)
            model_fn = model.forward

        latents = sample_fn(z_in, model_fn, **model_kwargs)[-1]
        if using_cfg:
            latents, _ = latents.chunk(2, dim=0)

        decoded = vae.decode(latents / 0.18215).sample
        images_t = torch.clamp(127.5 * decoded + 128.0, 0, 255)
        images_t = images_t.permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8)

    images = images_t.numpy()

    del model, vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Save images ----
    from PIL import Image
    for i, (img, out_path) in enumerate(zip(images, out_paths)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(str(out_path))
        print(f"Saved [{i+1}/{args.num_samples}]: {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
