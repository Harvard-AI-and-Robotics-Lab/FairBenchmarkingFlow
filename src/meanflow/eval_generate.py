"""Generate class-conditioned images from a MeanFlow checkpoint.

Generates num_samples images for a single ImageNet class using CFG and saves
each as a PNG. Uses the same generation pipeline as eval_imagenet.py.

Usage:
    python eval_generate.py \
        --ckpt_path weights/checkpoint_1200960 \
        --classname goldfish \
        --num_steps 1 \
        --cfg_scale 3.0 \
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

from meanflow import MeanFlow
from utils.logging_util import supress_checkpt_info
from utils.vae_util import LatentManager

supress_checkpt_info()
warnings.filterwarnings("ignore")

_MODEL_NAME = 'meanflow'

# ======================================================================
# Class validation
# ======================================================================

def _load_imagenet_classes():
    """Load imagenet_classes.json and return name_to_idx dict."""
    json_path = (Path(__file__).resolve().parent.parent
                 / 'utils' / 'classifiers' / 'siglip_classifier' / 'imagenet_classes.json')
    with open(json_path) as f:
        raw = json.load(f)
    name_to_idx = {v[1]: int(k) for k, v in raw.items()}
    return name_to_idx


def _resolve_class(classname, name_to_idx):
    if classname in name_to_idx:
        return name_to_idx[classname]
    # Try case-insensitive match
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
# Generation (mirrors eval_imagenet.py generate_with_cfg)
# ======================================================================

def _generate_with_cfg(variable, model, rng, n_sample, num_steps, img_size,
                       img_channels, labels, cfg_scale, num_classes):
    """Generate latents with classifier-free guidance."""
    import ml_collections
    config = ml_collections.ConfigDict()
    config.sampling = ml_collections.ConfigDict()
    config.sampling.num_steps = num_steps

    t_steps = model.apply({}, method=model.sampling_schedule())

    x_shape = (n_sample, img_size, img_size, img_channels)
    rng_xt, rng = jax.random.split(rng, 2)
    z_t = jax.random.normal(rng_xt, x_shape, dtype=model.dtype)

    null_labels = jnp.full((n_sample,), num_classes, dtype=jnp.int32)

    def cfg_step_fn(i, inputs):
        x_i, rng = inputs
        rng_step = jax.random.fold_in(rng, i)
        rng_cond, rng_uncond = jax.random.split(rng_step)

        t = t_steps[i]
        r = t_steps[i + 1]
        t_batch = jnp.repeat(t, n_sample)
        r_batch = jnp.repeat(r, n_sample)
        h = t_batch - r_batch

        u_cond = model.apply(
            variable, x_i, t=t_batch, h=h, y=labels, train=False,
            method=model.u_fn, rngs=dict(gen=rng_cond),
        )
        u_uncond = model.apply(
            variable, x_i, t=t_batch, h=h, y=null_labels, train=False,
            method=model.u_fn, rngs=dict(gen=rng_uncond),
        )

        u_guided = u_uncond + cfg_scale * (u_cond - u_uncond)
        x_next = x_i - jnp.einsum('n,n...->n...', h, u_guided)
        return (x_next, rng)

    outputs = jax.lax.fori_loop(0, num_steps, cfg_step_fn, (z_t, rng))
    return outputs[0]  # NHWC latents


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate images with MeanFlow')
    parser.add_argument('--ckpt_path', required=True, help='Path to MeanFlow checkpoint directory')
    parser.add_argument('--classname', required=True, help='ImageNet class name')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=3.0, help='CFG guidance scale')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output_path', default=None, help='Override output path')
    parser.add_argument('--model_cls', default='DiT_B_4', help='DiT model variant')
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

    # ---- Build model ----
    import ml_collections
    method_cfg = ml_collections.ConfigDict()
    method_cfg.noise_dist = 'logit_normal'
    method_cfg.P_mean = -0.4
    method_cfg.P_std = 1.0
    method_cfg.data_proportion = 0.75
    method_cfg.class_dropout_prob = 0.1
    method_cfg.guidance_eq = 'cfg'
    method_cfg.omega = args.cfg_scale
    method_cfg.kappa = 0.5
    method_cfg.t_start = 0.0
    method_cfg.t_end = 1.0
    method_cfg.norm_p = 1.0
    method_cfg.norm_eps = 0.01

    sampling_cfg = ml_collections.ConfigDict()
    sampling_cfg.seed = args.seed
    sampling_cfg.num_steps = args.num_steps
    sampling_cfg.schedule = 'default'
    sampling_cfg.sampling_timesteps = None
    sampling_cfg.num_classes = 1000

    model = MeanFlow(
        model_str=args.model_cls,
        model_config={},
        **sampling_cfg,
        **method_cfg,
    )

    # ---- Load checkpoint ----
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_path = str(Path(args.ckpt_path).resolve())
    print(f"Loading checkpoint from {ckpt_path}...")
    raw_ckpt = checkpointer.restore(ckpt_path)

    if not args.no_ema and 'ema_params' in raw_ckpt:
        params = raw_ckpt['ema_params']
        print('Using EMA params.')
    else:
        params = raw_ckpt['params']
        print('Using raw params.')
    del raw_ckpt

    params = jax.tree.map(lambda x: np.asarray(x), params)
    variable = {"params": params}

    # ---- Generate latents ----
    img_size = 32      # latent size (256 // 8)
    img_channels = 4
    num_classes = 1000

    labels = jnp.full((args.num_samples,), class_idx, dtype=jnp.int32)
    rng = jax.random.PRNGKey(args.seed)

    print(f"Generating {args.num_samples} image(s) for class '{args.classname}' "
          f"(idx={class_idx}), steps={args.num_steps}, cfg={args.cfg_scale}...")

    latents = _generate_with_cfg(
        variable, model, rng,
        n_sample=args.num_samples,
        num_steps=args.num_steps,
        img_size=img_size,
        img_channels=img_channels,
        labels=labels,
        cfg_scale=args.cfg_scale,
        num_classes=num_classes,
    )  # (N, H, W, C) NHWC

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
