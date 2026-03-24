"""Generate class-conditioned images from a SoFlow checkpoint.

Generates num_samples images for a single ImageNet class and saves each as a PNG.

SoFlow distills guidance into the model during training — a single forward pass
per step is needed with no separate unconditional pass. The --cfg_scale flag is
accepted for interface consistency but does not affect generation; the actual
guidance strength is fixed by the training config.

The --ckpt_path should be a weights directory containing:
    *.yaml          (training config, auto-discovered)
    ckpts/*.pt      (checkpoint file, latest by mtime wins)

Usage:
    python eval_generate.py \
        --ckpt_path weights/SoFlow/B-2-cond \
        --classname goldfish \
        --num_steps 1 \
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
import yaml
from PIL import Image

from models import Soflow

# Add src/ to import path for class validation helper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODEL_NAME = 'soflow'

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
# Weights directory discovery (mirrors eval_imagenet.py)
# ======================================================================

def _resolve_weights_dir(checkpoint_path):
    weights_dir = Path(checkpoint_path)
    if not weights_dir.is_dir():
        raise ValueError(f'--ckpt_path must be a directory, got: {checkpoint_path}')

    yaml_files = list(weights_dir.glob('*.yaml'))
    if not yaml_files:
        raise FileNotFoundError(f'No *.yaml config found in {weights_dir}')
    config_path = yaml_files[0]
    if len(yaml_files) > 1:
        print(f'[warn] Multiple YAML files found, using {config_path}')

    pt_files = list((weights_dir / 'ckpts').glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f'No ckpts/*.pt found under {weights_dir}')
    pt_path = max(pt_files, key=lambda p: p.stat().st_mtime)

    return config_path, pt_path


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate images with SoFlow')
    parser.add_argument('--ckpt_path', required=True,
                        help='SoFlow weights directory (contains *.yaml and ckpts/*.pt)')
    parser.add_argument('--classname', required=True, help='ImageNet class name')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='For record-keeping only — guidance is baked into the model')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output_path', default=None, help='Override output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_ema', action='store_true', help='Use raw params instead of EMA')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    # Resolve ckpt_path relative to script dir if relative
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parent / ckpt_path

    # Validate class name
    name_to_idx = _load_imagenet_classes()
    class_idx = _resolve_class(args.classname, name_to_idx)
    print(f"Class: '{args.classname}' → idx {class_idx}")

    # Build output paths
    out_paths = _make_output_paths(
        args.output_path, repo_root, args.classname,
        args.cfg_scale, args.num_steps, args.num_samples,
    )

    # ---- Discover config + checkpoint ----
    config_path, pt_path = _resolve_weights_dir(str(ckpt_path))
    print(f'Config:     {config_path}')
    print(f'Checkpoint: {pt_path}')

    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # ---- Build model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    data_channels = 4
    data_size = model_config.get('data_size', 32)
    num_classes = 1000
    noising_type = model_config.get('noising_type', 'Linear')
    coefficient_type = model_config.get('coefficient_type', 'Euler')
    model_type = model_config.get('model_type', 'DiT-B-4')
    cfg_drop_rate = model_config.get('cfg_drop_rate', 0.1)

    ema_model = Soflow(
        data_channels=data_channels,
        data_size=data_size,
        num_classes=num_classes,
        cfg_drop_rate=cfg_drop_rate,
        noising_type=noising_type,
        coefficient_type=coefficient_type,
        model_type=model_type,
    ).to(device)

    ckpt = torch.load(str(pt_path), map_location=device, weights_only=False)
    if not args.no_ema:
        ema_model.load_state_dict(ckpt['ema_model'])
        print('Using EMA params.')
    else:
        ema_model.load_state_dict(ckpt['model'])
        print('Using raw params.')
    del ckpt

    ema_model.requires_grad_(False)
    ema_model.eval()

    # ---- VAE ----
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device)
    vae.requires_grad_(False)
    vae.eval()

    # ---- Time schedule ----
    num_steps = args.num_steps
    eval_time = [1.0 - i / num_steps for i in range(num_steps + 1)]

    # ---- Generate ----
    labels = torch.full((args.num_samples,), class_idx, dtype=torch.long, device=device)

    print(f"Generating {args.num_samples} image(s) for class '{args.classname}' "
          f"(idx={class_idx}), steps={num_steps}...")

    with torch.inference_mode():
        x_t = torch.randn(
            (args.num_samples, data_channels, data_size, data_size), device=device,
        )
        t_ones = torch.ones((args.num_samples,), device=device)

        for i in range(len(eval_time) - 2):
            t_i = eval_time[i] * t_ones
            x_t = ema_model.solution_operator(x_t, t_i, torch.zeros_like(t_i), labels)
            next_t = eval_time[i + 1] * t_ones
            alpha_next = ema_model.scheduler.alpha(next_t).view(-1, 1, 1, 1)
            beta_next = ema_model.scheduler.beta(next_t).view(-1, 1, 1, 1)
            x_t = alpha_next * x_t + beta_next * torch.randn_like(x_t)

        # Final step
        t_last = eval_time[-2] * t_ones
        t_end_val = eval_time[-1] * t_ones  # = 0.0
        x_t = ema_model.solution_operator(x_t, t_last, t_end_val, labels)

        # Decode latents
        x_decoded = vae.decode(x_t / 0.18215).sample  # NCHW float
        samples = (x_decoded * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
        samples = samples.permute(0, 2, 3, 1).contiguous()  # NHWC

    images = samples.cpu().numpy()

    del ema_model, vae
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
