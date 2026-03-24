"""
Evaluate SiT on ImageNetV2 (matched-frequency) using shared benchmark utilities (ODE mode only).

Generates class-conditioned samples using all ImageNetV2 matched-frequency labels (in order),
decodes latents via VAE, and computes FID, Inception Score, CLIP Score, and
PickScore against real ImageNetV2 matched-frequency images.

CFG scale and number of sampling steps are set via --cfg and --num-steps.

Outputs are stored at:
  outputs/<model_name>/steps<N>/

Each run writes:
- metrics CSV
- sample grid PNG

Virtual environment example:
  python -m venv .venv
  . .venv/bin/activate
  pip install -r src/SiT/requirements.txt
"""

import argparse
import csv
import gc
import importlib
import os
import sys
from pathlib import Path

# Must be set before importing diffusers/transformers/huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.utils.data
from diffusers.models import AutoencoderKL
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image

from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler

# Use shared benchmark dataloader/eval utilities from src/utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataloader import ImageNetV2Dataset
from utils import eval as eval_metrics


def _preflight_eval_dependencies():
    """Check eval-time dependencies up-front so failures are actionable."""
    checks = [
        ("torchmetrics.image.fid", "torchmetrics[image]"),
        ("torchmetrics.image.inception", "torchmetrics[image]"),
        ("transformers", "transformers==4.43.4"),
        ("ImageReward", "ImageReward"),
        ("clip", "openai-clip"),
    ]
    missing = []
    for module_name, pip_name in checks:
        try:
            importlib.import_module(module_name)
        except Exception as err:
            missing.append((module_name, pip_name, str(err)))

    if missing:
        lines = ["Missing eval dependencies:"]
        for module_name, pip_name, err in missing:
            lines.append(f"  - import {module_name} (install: {pip_name})")
            lines.append(f"    error: {err}")
        lines.append("")
        lines.append(
            "Install with: python -m pip install -U "
            "\"torchmetrics[image]\" \"torchmetrics[multimodal]\" "
            "\"transformers==4.43.4\" \"huggingface_hub>=0.23.2,<1.0\" "
            "ImageReward openai-clip"
        )
        lines.append(
            "If 'clip' is still missing: "
            "python -m pip install -U git+https://github.com/openai/CLIP.git"
        )
        raise RuntimeError("\n".join(lines))


def _resolve_ckpt_path(args):
    if args.ckpt:
        ckpt = Path(os.path.expanduser(args.ckpt)).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return str(ckpt)

    default_dir = Path(__file__).resolve().parent / "pre_trained"
    if not default_dir.is_dir():
        raise FileNotFoundError(
            f"No --ckpt provided and default folder does not exist: {default_dir}"
        )

    candidates = []
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(default_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under {default_dir}")

    ckpt = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[info] Using checkpoint: {ckpt}")
    return str(ckpt)


def _load_sit_model(args, device):
    latent_size = args.image_size // 8
    state_dict = find_model(args.ckpt)

    # Try both variants to support both official and custom checkpoints.
    load_errors = []
    for learn_sigma in (False, True):
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
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


def _build_sample_fn(mode, args):
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    sampler = Sampler(transport)

    if mode == "ODE":
        return sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_sampling_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse,
        )

    return sampler.sample_sde(
        sampling_method=args.sampling_method,
        diffusion_form=args.diffusion_form,
        diffusion_norm=args.diffusion_norm,
        last_step=args.last_step,
        last_step_size=args.last_step_size,
        num_steps=args.num_sampling_steps,
    )


def _save_metrics_csv(metrics, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])


def _save_nfe_rolling_log(per_sample_nfe, csv_path, window=1000):
    """Rolling-window NFE statistics logged at every `window`-sample checkpoint.

    At each checkpoint k, the window covers [max(0, k-window+1), k] — rolling,
    not cumulative, so early-class and late-class regions are compared fairly.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    n = len(per_sample_nfe)
    checkpoints = list(range(0, n, window))
    if not checkpoints or checkpoints[-1] != n - 1:
        checkpoints.append(n - 1)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "window_start", "window_end",
                         "nfe_mean", "nfe_std", "nfe_min", "nfe_max"])
        for k in checkpoints:
            w_start = max(0, k - window + 1)
            subset = per_sample_nfe[w_start : k + 1]
            writer.writerow([
                k, w_start, k,
                f"{np.mean(subset):.4f}", f"{np.std(subset):.4f}",
                int(np.min(subset)), int(np.max(subset)),
            ])


def _save_nfe_per_class(per_sample_nfe, labels, class_names, csv_path):
    """Per-class NFE statistics: mean/std/min/max and sample count per ImageNet class."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    labels_arr = np.asarray(labels)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "nfe_mean", "nfe_std",
                         "nfe_min", "nfe_max", "n_samples"])
        for cls in sorted(np.unique(labels_arr)):
            mask = labels_arr == cls
            subset = per_sample_nfe[mask]
            name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            writer.writerow([
                int(cls), name,
                f"{np.mean(subset):.4f}", f"{np.std(subset):.4f}",
                int(np.min(subset)), int(np.max(subset)),
                int(np.sum(mask)),
            ])


def _save_sample_grid_to_path(images_uint8_nhwc, png_path, nrow=5, labels=None, class_names=None):
    from PIL import ImageDraw

    # Select 20 evenly-spaced images across the full set
    images_uint8_nhwc = np.asarray(images_uint8_nhwc)
    indices = np.linspace(0, len(images_uint8_nhwc) - 1, 20, dtype=int).tolist()
    selected = images_uint8_nhwc[indices]

    if labels is not None and class_names is not None:
        header_h = 20
        annotated = []
        for i, idx in enumerate(indices):
            img = Image.fromarray(selected[i])
            label_idx = int(labels[idx])
            text = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
            canvas = Image.new("RGB", (img.width, img.height + header_h), (20, 20, 20))
            canvas.paste(img, (0, header_h))
            draw = ImageDraw.Draw(canvas)
            max_chars = max(1, img.width // 6)
            draw.text((2, 3), text[:max_chars], fill=(255, 255, 255))
            annotated.append(np.array(canvas))
        selected = np.stack(annotated)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    img_tensors = torch.from_numpy(selected).permute(0, 3, 1, 2).float() / 255.0
    grid = make_grid(img_tensors, nrow=min(nrow, len(img_tensors)), padding=2)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(png_path)


def _sanitize_cfg_tag(cfg):
    if cfg == "natural":
        return "natural"
    if isinstance(cfg, float) and float(cfg).is_integer():
        cfg_str = str(int(cfg))
    else:
        cfg_str = str(cfg).replace(".", "p")
    return f"cfg{cfg_str}"


def _load_all_val_images(ref_dataset, batch_size, num_workers):
    """Load all ImageNetV2 images in dataset order as uint8 NHWC numpy array.

    Returns:
        real_images: (N, H, W, 3) uint8 numpy array.
        real_labels: (N,) int numpy array of class indices.
    """
    loader = torch.utils.data.DataLoader(
        ref_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    real_images = []
    real_labels = []
    for imgs, lbls in tqdm(loader, desc="Loading real images", leave=False):
        if isinstance(imgs, torch.Tensor):
            real_images.append(imgs.numpy())
        else:
            real_images.append(np.stack([np.array(img) for img in imgs]))
        real_labels.append(np.array(lbls))
    return np.concatenate(real_images, axis=0), np.concatenate(real_labels, axis=0)


def _generate_samples_memmap(
    model,
    vae,
    sample_fn,
    cfg_mode,
    cfg_value,
    val_labels,
    batch_size,
    latent_size,
    num_classes,
    image_size,
    out_prefix,
    seed,
    device,
):
    """Generate images conditioned on ordered ImageNetV2 labels, writing to memmap cache."""
    num_samples = len(val_labels)
    img_path = f"{out_prefix}_generated_images.npy"
    lbl_path = f"{out_prefix}_generated_labels.npy"
    nfe_path = f"{out_prefix}_per_sample_nfe.npy"
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        per_sample_nfe = np.load(nfe_path) if os.path.exists(nfe_path) else np.array([], dtype=np.int32)
        return np.load(img_path, mmap_mode="r"), np.load(lbl_path, mmap_mode="r"), per_sample_nfe

    torch.manual_seed(seed)
    mm_imgs = np.lib.format.open_memmap(
        img_path, mode="w+", dtype=np.uint8, shape=(num_samples, image_size, image_size, 3)
    )
    mm_lbls = np.lib.format.open_memmap(
        lbl_path, mode="w+", dtype=np.int64, shape=(num_samples,)
    )

    using_cfg = cfg_mode == "cfg"
    cursor = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    per_sample_nfe = np.zeros(num_samples, dtype=np.int32)

    with torch.inference_mode():
        for _ in tqdm(range(num_batches), desc="Generating", leave=False):
            cur_bs = min(batch_size, num_samples - cursor)
            z = torch.randn(cur_bs, model.in_channels, latent_size, latent_size, device=device)
            y = torch.from_numpy(val_labels[cursor:cursor + cur_bs]).to(device)

            if using_cfg:
                z_in = torch.cat([z, z], dim=0)
                y_null = torch.full((cur_bs,), num_classes, device=device, dtype=y.dtype)
                y_in = torch.cat([y, y_null], dim=0)
                model_kwargs = dict(y=y_in, cfg_scale=float(cfg_value))
                base_model_fn = model.forward_with_cfg
            else:
                z_in = z
                y_in = y
                model_kwargs = dict(y=y_in)
                base_model_fn = model.forward

            _nfe = [0]
            def _counted_fn(x, t, **kwargs):
                _nfe[0] += 1
                return base_model_fn(x, t, **kwargs)

            latents = sample_fn(z_in, _counted_fn, **model_kwargs)[-1]
            per_sample_nfe[cursor:cursor + cur_bs] = _nfe[0]
            if using_cfg:
                latents, _ = latents.chunk(2, dim=0)

            decoded = vae.decode(latents / 0.18215).sample
            images = torch.clamp(127.5 * decoded + 128.0, 0, 255)
            images = images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            mm_imgs[cursor:cursor + cur_bs] = images
            mm_lbls[cursor:cursor + cur_bs] = val_labels[cursor:cursor + cur_bs]
            cursor += cur_bs

    mm_imgs.flush()
    mm_lbls.flush()
    del mm_imgs, mm_lbls
    gc.collect()
    torch.cuda.empty_cache()
    np.save(nfe_path, per_sample_nfe)
    return np.load(img_path, mmap_mode="r"), np.load(lbl_path, mmap_mode="r"), per_sample_nfe


def _compute_metrics(generated_images, labels, real_images, class_names, metric_batch_size,
                     num_gpus, real_class_labels=None, compute_fid_IS_per_class=False):
    metrics = {}
    metrics["fid"] = float(
        eval_metrics.compute_fid(
            real_images,
            generated_images,
            batch_size=metric_batch_size,
            num_gpus=num_gpus,
        )
    )
    is_mean, is_std = eval_metrics.compute_inception_score(
        generated_images, batch_size=metric_batch_size, num_gpus=num_gpus
    )
    metrics["is_mean"] = float(is_mean)
    metrics["is_std"] = float(is_std)

    per_class = {}

    if compute_fid_IS_per_class and real_class_labels is not None:
        print("Computing per-class FID...")
        fid_pc = eval_metrics.compute_fid_per_class(
            real_images, real_class_labels, generated_images, labels,
            batch_size=metric_batch_size, num_gpus=num_gpus,
        )
        per_class["fid"] = {int(k): v for k, v in fid_pc.items()}
        print("Computing per-class IS...")
        is_pc = eval_metrics.compute_inception_score_per_class(
            generated_images, labels,
            batch_size=metric_batch_size, num_gpus=num_gpus,
        )
        per_class["inception_score"] = {int(k): v for k, v in is_pc.items()}

    clip_mean, clip_per_class = eval_metrics.compute_clip_score(
        generated_images,
        labels,
        class_names,
        batch_size=metric_batch_size,
        num_gpus=num_gpus,
        return_per_class=True,
    )
    metrics["clip_score"] = float(clip_mean)
    prompts = [f"a photo of a {class_names[int(lbl)]}" for lbl in labels]
    pick_mean, pick_per_class = eval_metrics.compute_pick_score(
        generated_images,
        prompts,
        batch_size=metric_batch_size,
        num_gpus=num_gpus,
        class_labels=labels,
        return_per_class=True,
    )
    metrics["pick_score"] = float(pick_mean)
    per_class["clip_score"] = {int(k): v for k, v in clip_per_class.items()}
    per_class["pick_score"] = {int(k): v for k, v in pick_per_class.items()}
    return metrics, per_class


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _preflight_eval_dependencies()

    # Parse cfg argument: float value or "natural"
    if args.cfg.lower() == "natural":
        cfg_mode, cfg_value = "natural", None
    else:
        cfg_mode, cfg_value = "cfg", float(args.cfg)

    steps = args.num_steps
    args.num_sampling_steps = steps + 1  # +1: integrator needs N+1 time points for N steps

    args.ckpt = _resolve_ckpt_path(args)
    model = _load_sit_model(args, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()

    repo_root = Path(__file__).resolve().parent.parent.parent
    outputs_root = Path(args.output_root) if args.output_root else (repo_root / "outputs" / args.model_name)
    outputs_root = outputs_root.resolve()
    cache_root = outputs_root / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    data_path = os.path.expanduser(args.data_path)
    ref_dataset = ImageNetV2Dataset(
        data_path=data_path,
        image_size=args.image_size,
        return_uint8=True,
    )
    val_labels_all = np.array([s[1] for s in ref_dataset.samples], dtype=np.int64)
    class_names = ref_dataset.class_names
    full_count = len(ref_dataset)
    print(f"[info] ImageNetV2 matched-frequency dataset size={full_count}")

    print("[info] Loading all real ImageNetV2 matched-frequency images...")
    real_images, real_labels = _load_all_val_images(
        ref_dataset,
        batch_size=args.real_batch_size,
        num_workers=args.num_workers,
    )
    print(f"[info] Real images loaded: {real_images.shape}")

    sample_fn = _build_sample_fn("ODE", args)

    step_dir = outputs_root / f"steps{steps}"
    step_dir.mkdir(parents=True, exist_ok=True)

    cfg_tag = _sanitize_cfg_tag("natural" if cfg_mode == "natural" else cfg_value)
    print(f"\n=== Running cfg={cfg_tag}, steps={steps} ===")

    run_stem = f"{args.model_name}_{cfg_tag}_steps{steps}_imagenetv2"
    run_subdir = step_dir / run_stem
    run_subdir.mkdir(parents=True, exist_ok=True)
    csv_path = run_subdir / f"{run_stem}.csv"
    png_path = run_subdir / f"{run_stem}_samples.png"
    run_cache_prefix = str(cache_root / run_stem)

    if args.resume and csv_path.exists():
        print(f"[skip] {csv_path} exists (resume enabled).")
    else:
        generated_images, labels, per_sample_nfe = _generate_samples_memmap(
            model=model,
            vae=vae,
            sample_fn=sample_fn,
            cfg_mode=cfg_mode,
            cfg_value=cfg_value,
            val_labels=val_labels_all,
            batch_size=args.gen_batch_size,
            latent_size=args.image_size // 8,
            num_classes=args.num_classes,
            image_size=args.image_size,
            out_prefix=run_cache_prefix,
            seed=args.seed,
            device=device,
        )

        metrics, per_class = _compute_metrics(
            generated_images=generated_images,
            labels=labels,
            real_images=real_images,
            class_names=class_names,
            metric_batch_size=args.metric_batch_size,
            num_gpus=args.metric_num_gpus,
            real_class_labels=real_labels,
            compute_fid_IS_per_class=args.compute_fid_IS_per_class,
        )
        metrics["num_samples"] = float(full_count)
        metrics["num_steps"] = float(steps)
        metrics["cfg_scale"] = "natural" if cfg_mode == "natural" else float(cfg_value)
        if len(per_sample_nfe) > 0:
            metrics["nfe_mean"] = float(np.mean(per_sample_nfe))
            metrics["nfe_std"] = float(np.std(per_sample_nfe))
            metrics["nfe_min"] = float(np.min(per_sample_nfe))
            metrics["nfe_max"] = float(np.max(per_sample_nfe))

        _save_metrics_csv(metrics, str(csv_path))

        if len(per_sample_nfe) > 0:
            nfe_rolling_path = str(run_subdir / f"{run_stem}_nfe_rolling.csv")
            _save_nfe_rolling_log(per_sample_nfe, nfe_rolling_path)
            nfe_class_path = str(run_subdir / f"{run_stem}_nfe_per_class.csv")
            _save_nfe_per_class(per_sample_nfe, labels, class_names, nfe_class_path)
            print(f"[done] nfe_rolling={nfe_rolling_path}")
            print(f"[done] nfe_per_class={nfe_class_path}")

        _save_sample_grid_to_path(
            generated_images,
            str(png_path),
            nrow=5,
            labels=labels,
            class_names=class_names,
        )

        per_class_csv = str(run_subdir / f"{run_stem}_per_class.csv")
        eval_metrics.save_per_class_metrics(per_class, per_class_csv, class_names)

        density_path = str(run_subdir / f"{run_stem}_per_class_density.png")
        eval_metrics.save_density_plots(per_class, density_path, class_names)

        print(f"[done] FID={metrics['fid']:.4f}  IS={metrics['is_mean']:.4f}±{metrics['is_std']:.4f}"
              f"  CLIP={metrics['clip_score']:.4f}  PickScore={metrics['pick_score']:.4f}")
        print(f"[done] csv={csv_path}")
        print(f"[done] png={png_path}")
        print(f"[done] per_class_csv={per_class_csv}")
        print(f"[done] density={density_path}")

    print("\nSiT ImageNetV2 benchmark run completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SiT ImageNetV2 benchmark (ODE mode) using src/utils/dataloader and src/utils/eval.py"
    )

    parser.add_argument("--data-path", type=str, default="~/autodl-tmp/imagenetv2")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to SiT checkpoint file. If omitted, uses latest file in src/SiT/pre_trained/")
    parser.add_argument("--model", type=str, default="SiT-XL/2", choices=list(SiT_models.keys()))
    parser.add_argument("--vae", type=str, default="mse", choices=["ema", "mse"])
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg", type=str, default="7.0",
                        help="CFG scale as a float (e.g. 7.0) or 'natural' to disable CFG")
    parser.add_argument("--num-steps", type=int, default=250,
                        help="Number of ODE sampling steps")
    parser.add_argument("--model-name", type=str, default="sit")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Defaults to repo_root/outputs/<model_name>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--real-batch-size", type=int, default=128)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--metric-num-gpus", type=int, default=0,
                        help="0 means all visible GPUs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip run if CSV already exists")
    parser.add_argument("--compute-fid-IS-per-class", dest="compute_fid_IS_per_class",
                        action="store_true", default=False,
                        help="Compute per-class FID and IS (expensive — off by default)")

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_args()
    main(args)
