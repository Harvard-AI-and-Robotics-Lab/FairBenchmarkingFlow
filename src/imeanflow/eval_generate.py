"""Generate class-conditioned images from an iMeanFlow checkpoint.

Generates num_samples images for a single ImageNet class using CFG and saves
each as a PNG. Uses the same generation pipeline as eval_imagenet.py.

Usage:
    python eval_generate.py \
        --ckpt_path path/to/checkpoint \
        --classname goldfish \
        --num_steps 1 \
        --cfg_scale 8.0 \
        --num_samples 4
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Flax classes are deprecated.*")
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from pathlib import Path
import json
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from jax import random

from imf import iMeanFlow, generate
from utils.ckpt_util import restore_checkpoint
from utils.lr_utils import lr_schedules
from utils.trainstate_util import create_train_state
from utils.vae_util import LatentManager

warnings.filterwarnings("ignore")

_MODEL_NAME = 'imeanflow'

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
    parser = argparse.ArgumentParser(description='Generate images with iMeanFlow')
    parser.add_argument('--ckpt_path', required=True, help='Path to iMeanFlow checkpoint directory')
    parser.add_argument('--classname', required=True, help='ImageNet class name')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=8.0, help='CFG guidance scale (omega)')
    parser.add_argument('--t_min', type=float, default=0.4, help='CFG interval lower bound')
    parser.add_argument('--t_max', type=float, default=0.65, help='CFG interval upper bound')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output_path', default=None, help='Override output path')
    parser.add_argument('--model_str', default='MiT_B_2', help='MiT model variant')
    parser.add_argument('--vae_type', default='mse', help='VAE variant (mse or ema)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_ema', action='store_true', help='Use raw params instead of EMA')
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

    # ---- Load config and build model ----
    from configs.default import get_config
    config = get_config()
    config.model.model_str = args.model_str
    config.dataset.vae = args.vae_type
    config.training.seed = args.seed
    config.sampling.num_steps = args.num_steps
    config.sampling.omega = args.cfg_scale
    config.sampling.t_min = args.t_min
    config.sampling.t_max = args.t_max
    config.fid.device_batch_size = args.num_samples
    config.load_from = args.ckpt_path
    config.eval_only = True

    model_config = config.model.to_dict()
    model = iMeanFlow(**model_config, eval=True)

    # ---- Load checkpoint ----
    rng = random.key(0)
    lr_fn = lr_schedules(config, 1000)
    state = create_train_state(rng, config, model, config.dataset.image_size, lr_fn)
    state = restore_checkpoint(state, args.ckpt_path)

    if not args.no_ema:
        params = state.ema_params
        print('Using EMA params.')
    else:
        params = state.params
        print('Using raw params.')
    del state

    params = jax.tree.map(lambda x: np.asarray(x), params)
    variable = {"params": params}

    # ---- Generate latents ----
    img_size = config.dataset.image_size  # 32

    labels = jnp.full((args.num_samples,), class_idx, dtype=jnp.int32)
    rng_gen = random.PRNGKey(args.seed)

    print(f"Generating {args.num_samples} image(s) for class '{args.classname}' "
          f"(idx={class_idx}), steps={args.num_steps}, cfg={args.cfg_scale}...")

    latents = generate(
        variable, model, rng_gen,
        args.num_samples,
        config,
        args.num_steps,
        args.cfg_scale,
        args.t_min,
        args.t_max,
        labels=labels,
    )  # NHWC latents

    del variable, params
    import gc; gc.collect()

    # ---- Decode latents via VAE ----
    print("Decoding latents with VAE...")
    latents_nchw = latents.transpose(0, 3, 1, 2)  # NHWC → NCHW
    latent_manager = LatentManager(args.vae_type, args.num_samples, img_size)
    decoded = latent_manager.decode(jnp.array(latents_nchw))  # (N, 3, H, W)
    decoded = decoded.transpose(0, 2, 3, 1)                    # → NHWC
    decoded = 127.5 * decoded + 128.0
    decoded = jnp.clip(decoded, 0, 255).astype(jnp.uint8)
    images = np.asarray(decoded)

    # ---- Save images ----
    for i, (img, out_path) in enumerate(zip(images, out_paths)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(str(out_path))
        print(f"Saved [{i+1}/{args.num_samples}]: {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
