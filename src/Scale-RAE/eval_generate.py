"""Generate class-conditioned images from a Scale-RAE checkpoint.

Generates num_samples images for a single ImageNet class and saves each as PNG.
Scale-RAE is an autoregressive model (Qwen2 LLM + DiT diffusion head) that
generates one image at a time via conditional text prompts.

Usage:
    python eval_generate.py \
        --ckpt_path nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B \
        --decoder_path decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt \
        --classname goldfish \
        --cfg_scale 7.0 \
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

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent))

from scale_rae.constants import IMAGE_TOKEN_INDEX
from scale_rae.conversation import conv_templates
from scale_rae.mm_utils import tokenizer_image_token
from scale_rae.model.builder import load_pretrained_model
from scale_rae.model.multimodal_decoder import MultimodalDecoder
from scale_rae.utils import disable_torch_init

_MODEL_NAME = 'scale_rae'

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
    parser = argparse.ArgumentParser(description='Generate images with Scale-RAE')
    parser.add_argument('--ckpt_path', required=True,
                        help='HuggingFace model ID or local path for Scale-RAE LLM '
                             '(e.g. nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B)')
    parser.add_argument('--decoder_path', required=True,
                        help='Path to RAE decoder .pt weights file')
    parser.add_argument('--classname', required=True, help='ImageNet class name')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Number of diffusion inference steps for the DiT head (default: 1)')
    parser.add_argument('--cfg_scale', type=float, default=7.0,
                        help='Guidance level (1.0 = natural, 7.0 = high guidance)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output_path', default=None, help='Override output path')
    parser.add_argument('--image_size', type=int, default=256, help='Output image resolution')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Max image tokens for autoregressive generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference')
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

    # ---- Load model ----
    disable_torch_init()
    print(f"Loading Scale-RAE model from {args.ckpt_path}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.ckpt_path,
        model_base=None,
        model_name=Path(args.ckpt_path).name,
        device=args.device,
    )

    # Override diffusion inference steps if requested
    if args.num_steps is not None and args.num_steps > 0 and hasattr(model, 'diff_head'):
        import math as _math
        from scale_rae.model.diffusion_loss.diffusion import create_diffusion as _create_diffusion

        dh = model.diff_head
        _input_dim = dh.diffusion_channels * dh.diffusion_tokens
        _ratio = _math.sqrt(dh.base_dim / _input_dim)

        dh.inference_flow = _create_diffusion(
            str(args.num_steps + 1),
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=False,
            rescale_learned_sigmas=False,
            diffusion_steps=1000,
            input_base_dimension_ratio=_ratio,
            diffusion_type="rf",
            use_loss_weighting=False,
            use_schedule_shift=True,
            diffusion_kwargs=None,
        )
        print(f"  Overrode diff_head.inference_flow to {args.num_steps} steps.")

    # ---- Load RAE decoder ----
    print(f"Loading RAE decoder from {args.decoder_path}...")
    decoder = MultimodalDecoder(
        pretrained_encoder_path="google/siglip2-so400m-patch14-224",
        general_decoder_config=str(_script_dir / "decoder"),
        num_patches=256,
        drop_cls_token=True,
        decoder_path=args.decoder_path,
    )
    decoder = decoder.to(model.device)
    if hasattr(decoder, 'image_mean'):
        decoder.image_mean = decoder.image_mean.to(model.device)
        decoder.image_std = decoder.image_std.to(model.device)

    # Special token IDs
    start_image_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    end_image_token_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    prompt = f"Can you generate a photo of a {args.classname}?"
    print(f"Prompt: '{prompt}'")
    print(f"Generating {args.num_samples} image(s), cfg={args.cfg_scale}...")

    # ---- Generate images (one at a time — model is autoregressive) ----
    from PIL import Image as PILImage

    images = []
    with torch.inference_mode():
        for i in range(args.num_samples):
            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)

            output_ids, image_embeds = model.generate(
                input_ids,
                images=None,
                output_image=True,
                do_sample=True,
                temperature=0.0,
                use_customize_greedy=True,
                top_p=None,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                start_image_token_id=start_image_token_id,
                end_image_token_id=end_image_token_id,
                eos_token_id=eos_token_id,
                guidance_level=args.cfg_scale,
            )

            if image_embeds is not None and image_embeds.ndim > 1:
                image_embeds_3d = image_embeds.unsqueeze(0)
                empty_cls = torch.zeros(
                    (1, 1, image_embeds_3d.shape[-1]),
                    device=image_embeds_3d.device,
                    dtype=image_embeds_3d.dtype,
                )
                image_features = torch.cat([empty_cls, image_embeds_3d], dim=1)

                with torch.no_grad():
                    xs_recon = decoder(image_features)  # [1, C, H, W]
                xs_recon = xs_recon.permute(0, 2, 3, 1).clip(0, 1)
                img_np = (xs_recon.cpu().numpy() * 255).astype('uint8')[0]
            else:
                print(f"\n  Warning: no image generated for sample {i}, using blank frame.")
                img_np = np.zeros((256, 256, 3), dtype=np.uint8)

            img = PILImage.fromarray(img_np)
            img_resized = img.resize((args.image_size, args.image_size), PILImage.LANCZOS)
            images.append(np.array(img_resized))

    del model, tokenizer, decoder, image_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Save images ----
    for i, (img, out_path) in enumerate(zip(images, out_paths)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(img).save(str(out_path))
        print(f"Saved [{i+1}/{args.num_samples}]: {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
