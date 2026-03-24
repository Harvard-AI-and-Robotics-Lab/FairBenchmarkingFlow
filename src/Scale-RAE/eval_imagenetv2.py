"""Evaluate Scale-RAE on ImageNetV2 (matched-frequency).

Generates ImageNetV2 class-conditioned samples using text prompts
("Can you generate a photo of a {class_name}?") via the Scale-RAE
LLM-based autoregressive image generation model, and computes
FID, Inception Score, CLIP Score, and PickScore against real
ImageNetV2 matched-frequency images using the shared evaluation utilities.

Scale-RAE is an autoregressive model (Qwen2 LLM + DiT diffusion head).
The --num_steps argument is accepted for benchmark format compatibility
but has no effect on generation — the model always performs its full
autoregressive decoding pass with max_new_tokens image tokens.

The guidance parameter is --guidance_level (not cfg_scale). Typical
values: 1.0 (natural/default), 7.0 (high guidance).

Follows the same phase structure as eval_imagenet.py:
    Phase 1: Generate images (Scale-RAE pipeline, single GPU)
    Phase 2: Load real ImageNetV2 references (shared dataloader)
    Phase 3: Compute metrics (shared eval)
    Phase 4: Save results

Usage:
    # 5-sample sanity check
    python eval_imagenetv2.py \\
        --data_path /path/to/ImageNetV2 \\
        --decoder_path decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt \\
        --num_samples 5 --guidance_level 7.0 --model_name scale_rae

    # Full run (natural guidance)
    python eval_imagenetv2.py \\
        --data_path /path/to/ImageNetV2 \\
        --decoder_path decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt \\
        --guidance_level 1.0 --model_name natural_scale_rae
"""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image as PILImage
from tqdm import tqdm

# ---- Path setup ----
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))          # for scale_rae package
sys.path.insert(0, str(_script_dir.parent))   # for utils.dataloader / utils.eval

from utils.dataloader import ImageNetV2Dataset
from utils import eval as eval_metrics

from scale_rae.constants import IMAGE_TOKEN_INDEX
from scale_rae.conversation import conv_templates
from scale_rae.mm_utils import tokenizer_image_token
from scale_rae.model.builder import load_pretrained_model
from scale_rae.model.multimodal_decoder import MultimodalDecoder
from scale_rae.utils import disable_torch_init


# ======================================================================
# Metrics helpers
# ======================================================================

def _save_best_worst_fid_grid(generated_images, all_labels, fid_per_class,
                               class_names, save_path):
    """Save a 2x10 image grid showing the 10 best and 10 worst per-class FID classes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_labels_arr = np.asarray(all_labels)
    sorted_classes = sorted(fid_per_class.items(), key=lambda x: x[1])
    best10  = sorted_classes[:10]
    worst10 = sorted_classes[-10:]

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for row_idx, (group, color) in enumerate([(best10, '#1a9850'), (worst10, '#d73027')]):
        for col_idx, (cls, fid_val) in enumerate(group):
            ax = axes[row_idx, col_idx]
            match = np.where(all_labels_arr == cls)[0]
            if len(match):
                img = generated_images[match[0]]
            else:
                img = np.zeros((generated_images.shape[1], generated_images.shape[2], 3),
                               dtype=np.uint8)
            ax.imshow(img)
            ax.axis('off')
            name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            ax.set_title(f'{name}\n{fid_val:.2f}', fontsize=7, color=color, pad=2)

    axes[0, 0].set_ylabel('Best 10\n(↓ FID)', fontsize=9, color='#1a9850', labelpad=6)
    axes[1, 0].set_ylabel('Worst 10\n(↑ FID)', fontsize=9, color='#d73027', labelpad=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def _compute_all_metrics(generated_images, all_labels, real_images, class_names,
                         real_class_labels=None, compute_fid_IS_per_class=False,
                         cache_dir=None):
    """Compute all evaluation metrics and return as (global_dict, per_class_dict).

    When *cache_dir* is provided each metric is saved to a JSON file as
    soon as it finishes.  On a subsequent call with the same *cache_dir*,
    already-cached metrics are loaded instead of recomputed.
    """
    cache_path = os.path.join(cache_dir, 'metrics_cache.json') if cache_dir else None
    cached = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)

    _per_class_cache_keys = ('clip_score_per_class', 'pick_score_per_class',
                             'fid_per_class', 'is_per_class')

    def _cache(results, per_class):
        if cache_path:
            data = dict(results)
            data['clip_score_per_class'] = per_class.get('clip_score', {})
            data['pick_score_per_class'] = per_class.get('pick_score', {})
            data['fid_per_class'] = per_class.get('fid', {})
            data['is_per_class'] = per_class.get('inception_score', {})
            with open(cache_path, 'w') as f:
                json.dump(data, f)

    results = {k: v for k, v in cached.items() if k not in _per_class_cache_keys}
    per_class = {}

    if 'fid' not in results:
        print('Computing FID...')
        results['fid'] = float(eval_metrics.compute_fid(real_images, generated_images))
        _cache(results, per_class)
    else:
        print(f'FID: cached ({results["fid"]:.4f})')

    if 'is_mean' not in results:
        print('Computing Inception Score...')
        is_mean, is_std = eval_metrics.compute_inception_score(generated_images)
        results['is_mean'] = float(is_mean)
        results['is_std'] = float(is_std)
        _cache(results, per_class)
    else:
        print(f'IS: cached ({results["is_mean"]:.4f})')

    if compute_fid_IS_per_class and real_class_labels is not None:
        _expected_classes = set(int(c) for c in np.unique(all_labels))

        _cached_fid_classes = set(int(k) for k in cached.get('fid_per_class', {}).keys())
        if not _expected_classes.issubset(_cached_fid_classes):
            print('Computing per-class FID...')
            fid_pc = eval_metrics.compute_fid_per_class(
                real_images, real_class_labels, generated_images, all_labels
            )
            per_class['fid'] = {str(k): v for k, v in fid_pc.items()}
            _cache(results, per_class)
        else:
            print('FID per-class: cached')
            per_class['fid'] = {str(k): v for k, v in cached['fid_per_class'].items()}

        _cached_is_classes = set(int(k) for k in cached.get('is_per_class', {}).keys())
        if not _expected_classes.issubset(_cached_is_classes):
            print('Computing per-class IS...')
            is_pc = eval_metrics.compute_inception_score_per_class(
                generated_images, all_labels
            )
            per_class['inception_score'] = {str(k): v for k, v in is_pc.items()}
            _cache(results, per_class)
        else:
            print('IS per-class: cached')
            per_class['inception_score'] = {str(k): v for k, v in cached['is_per_class'].items()}

    if 'clip_score' not in results:
        print('Computing CLIP Score...')
        clip_mean, clip_per_class = eval_metrics.compute_clip_score(
            generated_images, all_labels, class_names, return_per_class=True
        )
        results['clip_score'] = float(clip_mean)
        per_class['clip_score'] = {str(k): v for k, v in clip_per_class.items()}
        _cache(results, per_class)
    else:
        print(f'CLIP Score: cached ({results["clip_score"]:.4f})')
        per_class['clip_score'] = {str(k): v for k, v in cached.get('clip_score_per_class', {}).items()}

    prompts = [f"a photo of a {class_names[l]}" for l in all_labels]

    if 'pick_score' not in results:
        print('Computing PickScore...')
        pick_mean, pick_per_class = eval_metrics.compute_pick_score(
            generated_images, prompts, class_labels=all_labels, return_per_class=True
        )
        results['pick_score'] = float(pick_mean)
        per_class['pick_score'] = {str(k): v for k, v in pick_per_class.items()}
        _cache(results, per_class)
    else:
        print(f'PickScore: cached ({results["pick_score"]:.4f})')
        per_class['pick_score'] = {str(k): v for k, v in cached.get('pick_score_per_class', {}).items()}

    return results, per_class


# ======================================================================
# Phase 1: Image Generation
# ======================================================================

def generate_images(model_path, decoder_path, num_samples, guidance_level,
                    image_size, class_names, val_labels,
                    max_new_tokens, ckpt_every, output_dir, device,
                    num_steps=None, shard_id=0):
    """Generate images using Scale-RAE autoregressive pipeline."""
    # ---- Load model ----
    disable_torch_init()
    print(f"Loading Scale-RAE model from {model_path}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=Path(model_path).name,
        device=device,
    )

    # ---- Override diffusion inference steps if requested ----
    if num_steps is not None and num_steps > 0 and hasattr(model, 'diff_head'):
        import math as _math
        from scale_rae.model.diffusion_loss.diffusion import create_diffusion as _create_diffusion

        dh = model.diff_head
        _input_dim = dh.diffusion_channels * dh.diffusion_tokens
        _ratio = _math.sqrt(dh.base_dim / _input_dim)

        dh.inference_flow = _create_diffusion(
            str(num_steps + 1),  # +1: space_timesteps gives N indices → N-1 actual denoising steps
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
        print(f"  Overrode diff_head.inference_flow to {num_steps} steps "
              f"(was 50 hardcoded). "
              f"used_timesteps count={len(dh.inference_flow.used_timesteps)}")
    else:
        print(f"  Using default diff_head inference steps (50 hardcoded).")

    # ---- Load RAE decoder ----
    print(f"Loading RAE decoder from {decoder_path}...")
    decoder = MultimodalDecoder(
        pretrained_encoder_path="google/siglip2-so400m-patch14-224",
        general_decoder_config=str(_script_dir / "decoder"),
        num_patches=256,
        drop_cls_token=True,
        decoder_path=decoder_path,
    )
    decoder = decoder.to(model.device)
    if hasattr(decoder, 'image_mean'):
        decoder.image_mean = decoder.image_mean.to(model.device)
        decoder.image_std = decoder.image_std.to(model.device)

    # ---- Special token IDs (constant for all generations) ----
    start_image_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    end_image_token_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # ---- Check for partial checkpoint ----
    os.makedirs(output_dir, exist_ok=True)
    partial_path = os.path.join(output_dir, f'_partial_checkpoint_shard{shard_id}.npz')

    all_images = []
    all_labels = []
    num_generated = 0

    if os.path.exists(partial_path):
        print("Resuming from partial checkpoint...")
        data = np.load(partial_path)
        all_images = list(data['images'])
        all_labels = list(data['labels'].tolist())
        num_generated = int(data['num_generated'])
        print(f"  Resumed: {num_generated}/{num_samples} images")

    # ---- Label sequence (ordered by the v2 dataset) ----
    remaining_labels = val_labels[num_generated:]

    pbar = tqdm(total=num_samples, initial=num_generated, desc="Generating")
    batch_count = 0

    with torch.inference_mode():
        for label in remaining_labels:
            label = int(label)
            class_name = class_names[label]
            prompt = f"Can you generate a photo of a {class_name}?"

            # Build conversation
            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)

            # Generate latent embeddings autoregressively
            output_ids, image_embeds = model.generate(
                input_ids,
                images=None,
                output_image=True,
                do_sample=True,
                temperature=0.0,
                use_customize_greedy=True,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                start_image_token_id=start_image_token_id,
                end_image_token_id=end_image_token_id,
                eos_token_id=eos_token_id,
                guidance_level=guidance_level,
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
                print(f"\n  Warning: no image generated for class '{class_name}', "
                      f"using blank frame.")
                img_np = np.zeros((256, 256, 3), dtype=np.uint8)

            # Resize to target resolution
            img = PILImage.fromarray(img_np)
            img_resized = img.resize((image_size, image_size), PILImage.LANCZOS)
            all_images.append(np.array(img_resized))
            all_labels.append(label)
            num_generated += 1
            batch_count += 1
            pbar.update(1)

            # Periodic checkpoint
            if ckpt_every > 0 and batch_count % ckpt_every == 0:
                np.savez(
                    partial_path,
                    images=np.stack(all_images),
                    labels=np.array(all_labels, dtype=np.int64),
                    num_generated=np.array(num_generated),
                )
                print(f"\n  Checkpoint: {num_generated}/{num_samples}")

    pbar.close()

    generated_images = np.stack(all_images)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    del model, tokenizer, decoder, image_processor
    gc.collect()
    torch.cuda.empty_cache()

    return generated_images, all_labels_arr

# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Scale-RAE on ImageNetV2 matched-frequency (FID, IS, CLIP Score, PickScore)"
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNetV2 root directory (contains imagenetv2-matched-frequency-format-val/)')
    parser.add_argument('--model_path', type=str,
                        default='nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B',
                        help='HuggingFace model ID or local path for Scale-RAE LLM')
    parser.add_argument('--decoder_path', type=str, required=True,
                        help='Path to local RAE decoder .pt weights file '
                             '(e.g. decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Override number of samples (for sanity checks)')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Number of diffusion inference steps for the DiT head')
    parser.add_argument('--guidance_level', type=float, default=1.0,
                        help='Guidance level for Scale-RAE generation '
                             '(default 1.0 = natural; try 7.0 for high guidance)')
    parser.add_argument('--model_name', type=str, default='scale_rae',
                        help='Name used for output CSV and PNG tag')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Max image tokens for autoregressive generation (default 512)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for class label sampling')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Target image resolution for metrics (default 256)')
    parser.add_argument('--ckpt_every', type=int, default=100,
                        help='Save partial checkpoint every N images (0 to disable)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (default: cuda:0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to generated_checkpoint.npz to skip generation')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='GPU/shard index (0-indexed)')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of parallel shards')
    parser.add_argument('--gen_only', action='store_true',
                        help='Only generate images (skip metrics). '
                             'Used for parallel shards.')
    parser.add_argument('--compute_fid_IS_per_class', action='store_true', default=False,
                        help='Compute per-class FID and IS (expensive — off by default)')
    args = parser.parse_args()

    # ---- Determine number of samples ----
    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        ds = ImageNetV2Dataset(data_path=args.data_path,
                               image_size=args.image_size, return_uint8=True)
        num_samples = len(ds)
        del ds

    print("Scale-RAE ImageNetV2 Evaluation")
    print(f"  Model:         {args.model_path}")
    print(f"  Decoder:       {args.decoder_path}")
    print(f"  Samples:       {num_samples}")
    print(f"  Guidance:      {args.guidance_level}")
    print(f"  num_steps:     {args.num_steps}")
    print(f"  Model name:    {args.model_name}")
    print(f"  Shard:         {args.shard_id}/{args.num_shards}")
    print("=" * 60)

    # ---- Output directory ----
    repo_root = Path(__file__).resolve().parent.parent.parent
    tag = args.model_name
    subfolder_name = f'{tag}_steps_{args.num_steps}_imagenetv2'
    output_dir = repo_root / 'outputs' / 'scale_rae' / f'steps_{args.num_steps}'
    plot_subdir = output_dir / subfolder_name
    ckpt_dir = plot_subdir / '_checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Auto-merge shards if they exist (runs before resume/generate)
    # ==================================================================
    shard_files = sorted(ckpt_dir.glob('shard_*.npz'))
    if shard_files and not args.resume and not args.gen_only:
        print(f"Found {len(shard_files)} shard files. Merging...")
        shard_images, shard_labels = [], []
        for sf in shard_files:
            d = np.load(sf)
            shard_images.append(d['images'])
            shard_labels.append(d['labels'])
        merged_images = np.concatenate(shard_images, axis=0)[:num_samples]
        merged_labels = np.concatenate(shard_labels, axis=0)[:num_samples]
        print(f"Merged: {merged_images.shape}")

        ckpt_path = ckpt_dir / 'generated_checkpoint.npz'
        np.savez_compressed(str(ckpt_path),
                            images=merged_images, labels=merged_labels)
        print(f"Saved merged checkpoint to {ckpt_path}")

        args.resume = str(ckpt_path)
        del merged_images, merged_labels
        gc.collect()

    # ==================================================================
    # Phase 1: Generate images (or resume from checkpoint)
    # ==================================================================
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = np.load(args.resume)
        generated_images = ckpt['images'][:num_samples]
        all_labels = ckpt['labels'][:num_samples]
        del ckpt
        print(f"Loaded {generated_images.shape[0]} images from checkpoint.")
    else:
        print("=" * 60)
        print("Phase 1: Generating images with Scale-RAE...")
        print("=" * 60)

        ref_ds = ImageNetV2Dataset(data_path=args.data_path,
                                   image_size=args.image_size, return_uint8=True)
        class_names = ref_ds.class_names
        val_labels_all = np.array([s[1] for s in ref_ds.samples], dtype=np.int32)
        del ref_ds

        # ---- Shard the labels across GPUs ----
        all_val_labels = val_labels_all[:num_samples]
        shard_size = (num_samples + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, num_samples)
        shard_labels = all_val_labels[start_idx:end_idx]
        shard_n = len(shard_labels)

        print(f"  Shard {args.shard_id}/{args.num_shards}: "
              f"samples [{start_idx}:{end_idx}] ({shard_n} images) "
              f"on {args.device}")

        generated_images, all_labels = generate_images(
            model_path=args.model_path,
            decoder_path=args.decoder_path,
            num_samples=shard_n,
            guidance_level=args.guidance_level,
            image_size=args.image_size,
            class_names=class_names,
            val_labels=shard_labels,
            max_new_tokens=args.max_new_tokens,
            ckpt_every=args.ckpt_every,
            output_dir=str(ckpt_dir),
            device=args.device,
            num_steps=args.num_steps,
            shard_id=args.shard_id,
        )

        print(f"Phase 1 done (shard {args.shard_id}): {generated_images.shape}")

        shard_path = ckpt_dir / f'shard_{args.shard_id}.npz'
        np.savez_compressed(str(shard_path),
                            images=generated_images, labels=all_labels)
        print(f"Shard saved to {shard_path}")

        if args.gen_only:
            print(f"Shard {args.shard_id} done. Exiting (--gen_only).")
            return

        ckpt_path = ckpt_dir / 'generated_checkpoint.npz'
        np.savez_compressed(str(ckpt_path),
                            images=generated_images, labels=all_labels)

    # ==================================================================
    # Phase 2: Load real ImageNetV2 matched-frequency reference images
    # ==================================================================
    print("=" * 60)
    print("Phase 2: Loading real ImageNetV2 matched-frequency reference images...")
    print("=" * 60)

    ref_dataset = ImageNetV2Dataset(
        data_path=args.data_path,
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

    # ==================================================================
    # Phase 3: Compute metrics (with per-metric caching on resume)
    # ==================================================================
    print("=" * 60)
    print("Phase 3: Computing metrics...")
    print("=" * 60)

    metrics, per_class_metrics = _compute_all_metrics(
        generated_images, all_labels, real_images, class_names,
        real_class_labels=real_labels,
        compute_fid_IS_per_class=args.compute_fid_IS_per_class,
        cache_dir=str(ckpt_dir),
    )
    metrics.update({
        'num_samples': float(num_samples),
        'guidance_level': float(args.guidance_level),
        'num_steps': float(args.num_steps),
    })

    # ==================================================================
    # Phase 4: Save results
    # ==================================================================
    print("=" * 60)
    print("Phase 4: Saving results...")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_subdir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_subdir / f'{subfolder_name}.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])
    print(f"Metrics saved to {csv_path}")

    png_path = eval_metrics.save_sample_grid(
        generated_images, str(output_dir), subfolder_name,
        labels=all_labels, class_names=class_names,
    )
    print(f"Sample grid saved to {png_path}")

    per_class_int = {m: {int(k): v for k, v in d.items()} for m, d in per_class_metrics.items()}
    per_class_csv = str(plot_subdir / f'{subfolder_name}_per_class.csv')
    eval_metrics.save_per_class_metrics(per_class_int, per_class_csv, class_names)
    print(f"Per-class metrics saved to {per_class_csv}")

    density_path = str(plot_subdir / f'{subfolder_name}_per_class_density.png')
    eval_metrics.save_density_plots(per_class_int, density_path, class_names)
    print(f"Density plot saved to {density_path}")

    if per_class_metrics.get('fid'):
        fid_grid_path = str(plot_subdir / f'{subfolder_name}_best_worst_fid.png')
        _save_best_worst_fid_grid(
            generated_images, all_labels,
            {int(k): v for k, v in per_class_metrics['fid'].items()},
            class_names, fid_grid_path,
        )
        print(f'Best/worst FID grid saved to {fid_grid_path}')

    print('=' * 60)
    print(f'FID:          {metrics["fid"]:.4f}')
    print(f'IS:           {metrics["is_mean"]:.4f} +/- {metrics["is_std"]:.4f}')
    print(f'CLIP Score:   {metrics["clip_score"]:.4f}')
    print(f'PickScore:    {metrics["pick_score"]:.4f}')
    print(f'Guidance:     {args.guidance_level}')
    print(f'Steps:        {args.num_steps}')
    print(f'Samples:      {num_samples}')
    print(f'Output:       {csv_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
