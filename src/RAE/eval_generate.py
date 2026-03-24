"""Generate class-conditioned images from a RAE checkpoint.

Generates num_samples images for a single ImageNet class and saves each as PNG.
Uses a single GPU (no DDP) — simpler than eval_imagenet.py which uses torchrun.

Requires a YAML sampling config (--config) that points to the Stage 1 RAE
decoder and Stage 2 diffusion model weights.

Usage:
    python eval_generate.py \
        --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
        --ckpt_path path/to/stage2/checkpoint.pt \
        --classname goldfish \
        --num_steps 50 \
        --cfg_scale 1.5 \
        --num_samples 4
"""

import gc
import json
import math
import os
import sys
import warnings
from pathlib import Path
import argparse

import numpy as np
import torch

# ======================================================================
# Path setup (mirrors eval_imagenet.py to handle RAE's local utils/ conflict)
# ======================================================================
_script_dir = Path(__file__).resolve().parent
_rae_src = str(_script_dir / 'src')
_shared_src = str(_script_dir.parent)

sys.path.insert(0, _rae_src)

from omegaconf import OmegaConf
from stage1 import RAE
from stage2.transport import create_transport, Sampler
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs

# Clear RAE's utils from cache so shared utils can be imported
for _mod_name in list(sys.modules):
    if _mod_name == 'utils' or _mod_name.startswith('utils.'):
        del sys.modules[_mod_name]

sys.path.insert(0, _shared_src)

if _rae_src not in sys.path:
    sys.path.append(_rae_src)

warnings.filterwarnings("ignore")

_MODEL_NAME = 'rae'

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
    parser = argparse.ArgumentParser(description='Generate images with RAE')
    parser.add_argument('--config', required=True,
                        help='Path to RAE sampling YAML config (e.g. configs/stage2/sampling/...)')
    parser.add_argument('--ckpt_path', default=None,
                        help='Path to Stage 2 checkpoint .pt (optional override over config default)')
    parser.add_argument('--classname', required=True, help='ImageNet class name')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Number of ODE/SDE steps (overrides config default)')
    parser.add_argument('--cfg_scale', type=float, default=None,
                        help='CFG guidance scale (overrides config default)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output_path', default=None, help='Override output path')
    parser.add_argument('--precision', default='bf16', choices=['fp32', 'bf16'],
                        help='Model precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    # Validate class name
    name_to_idx = _load_imagenet_classes()
    class_idx = _resolve_class(args.classname, name_to_idx)
    print(f"Class: '{args.classname}' → idx {class_idx}")

    # ---- Parse config ----
    cfg = OmegaConf.load(args.config)
    (rae_config, model_config, transport_config, sampler_config,
     guidance_config, misc, _, _) = parse_configs(cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 entries.")

    misc = {} if misc is None else dict(misc)

    # Latent space
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    # Determine effective num_steps for output path naming
    eff_num_steps = args.num_steps
    if eff_num_steps is None and sampler_config is not None:
        sp = dict(sampler_config.get("params", {}))
        eff_num_steps = sp.get("num_steps", sp.get("steps", 50))
    eff_cfg = args.cfg_scale
    if eff_cfg is None and guidance_config is not None:
        eff_cfg = dict(guidance_config).get("scale", 1.0)

    # Build output paths
    out_paths = _make_output_paths(
        args.output_path, repo_root, args.classname,
        eff_cfg, eff_num_steps, args.num_samples,
    )

    # ---- Build models ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    use_bf16 = args.precision == "bf16"
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    print("Loading RAE (Stage 1 decoder) and diffusion model (Stage 2)...")
    rae: RAE = instantiate_from_config(rae_config).to(device)
    model = instantiate_from_config(model_config).to(device)

    # Override Stage 2 checkpoint if provided
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('ema', ckpt.get('model', ckpt))
        model.load_state_dict(state_dict)
        print(f"Loaded Stage 2 checkpoint from {args.ckpt_path}")

    rae.eval()
    model.eval()

    # ---- Transport + sampler ----
    transport_params = {}
    if transport_config is not None:
        transport_params = dict(transport_config.get("params", {}))
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)

    sampler_config_dict = {} if sampler_config is None else dict(sampler_config)
    sampler_mode = sampler_config_dict.get("mode", "ODE")
    sampler_params = dict(sampler_config_dict.get("params", {}))
    if args.num_steps is not None:
        sampler_params["num_steps"] = args.num_steps + 1  # integrator needs N+1 points for N steps

    mode = sampler_mode.upper()
    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode: {sampler_mode}")

    # ---- Guidance ----
    guidance_config_dict = {} if guidance_config is None else dict(guidance_config)
    guidance_scale = guidance_config_dict.get("scale", 1.0)
    if args.cfg_scale is not None:
        guidance_scale = args.cfg_scale
    guidance_method = guidance_config_dict.get("method", "cfg")
    t_min = guidance_config_dict.get("t_min", guidance_config_dict.get("t-min", 0.0))
    t_max = guidance_config_dict.get("t_max", guidance_config_dict.get("t-max", 1.0))

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    using_cfg = guidance_scale > 1.0

    print(f"Generating {args.num_samples} image(s) for class '{args.classname}' "
          f"(idx={class_idx}), steps={eff_num_steps}, cfg={guidance_scale}, mode={mode}...")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.no_grad():
        labels = torch.full((args.num_samples,), class_idx, dtype=torch.long, device=device)
        z = torch.randn(args.num_samples, *latent_size, device=device)

        if using_cfg:
            null_labels = torch.full((args.num_samples,), null_label, dtype=torch.long, device=device)

            def model_fn(x, t):
                with torch.autocast('cuda', **autocast_kwargs):
                    cond = model(x, t, labels)
                    uncond = model(x, t, null_labels)
                return uncond + guidance_scale * (cond - uncond)
        else:
            def model_fn(x, t):
                with torch.autocast('cuda', **autocast_kwargs):
                    return model(x, t, labels)

        samples = sample_fn(model_fn)(z)  # (N, C, H, W) latents in latent space

        # Decode via RAE Stage 1 decoder
        print("Decoding with RAE decoder...")
        with torch.autocast('cuda', **autocast_kwargs):
            decoded = rae.decode(samples)  # (N, 3, H, W) in [-1, 1]

    # Convert to uint8 NHWC
    decoded_np = decoded.float().cpu().numpy()  # (N, 3, H, W)
    decoded_np = decoded_np.transpose(0, 2, 3, 1)  # NHWC
    decoded_np = (decoded_np * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    del model, rae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Save images ----
    from PIL import Image
    for i, (img, out_path) in enumerate(zip(decoded_np, out_paths)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(str(out_path))
        print(f"Saved [{i+1}/{args.num_samples}]: {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
