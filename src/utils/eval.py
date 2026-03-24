"""Evaluation metrics for generative models.

Metrics: FID, Inception Score, CLIP Score, PickScore, and ImageReward.

All image inputs are expected as **uint8 [0, 255] NHWC numpy arrays**.
Uses torchmetrics for FID/IS/CLIP Score (standard-benchmark-compatible),
PickScore_v1 for PickScore, and ImageReward-v1.0 for ImageReward.

Multi-GPU: All metrics use torch.nn.DataParallel when multiple GPUs are
available, scaling batch size proportionally to GPU count.

Usage:
    import eval as eval_metrics

    fid = eval_metrics.compute_fid(real_images, generated_images)
    is_mean, is_std = eval_metrics.compute_inception_score(generated_images)
    clip = eval_metrics.compute_clip_score(images, labels, class_names)
    pick = eval_metrics.compute_pick_score(images, prompts)
    img_reward = eval_metrics.compute_image_reward(images, prompts)

    eval_metrics.save_metrics({'fid': fid, 'is_mean': is_mean}, output_dir, model_name)
    eval_metrics.save_sample_grid(sample_images, output_dir, model_name)
"""

import csv
import gc
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ======================================================================
# Helper
# ======================================================================

def _free_gpu():
    """Force-free GPU memory between metric computations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gpu_setup(device=None, num_gpus=0, min_free_mb=4096):
    """Return (device, device_ids) for multi-GPU support.

    Filters out GPUs with less than min_free_mb MiB of free memory to avoid
    OOM errors when other processes are occupying a GPU.

    Args:
        device: torch device string (auto-detected if None).
        num_gpus: Max number of GPUs to use. 0 = all available.
        min_free_mb: Minimum free GPU memory (MiB) required to include a GPU.
                     Default 4096 (4 GiB) to safely exclude GPUs that are
                     mostly occupied by other processes on shared clusters.

    Returns:
        (device, device_ids): Primary torch.device and sorted list of usable
                              GPU indices. Returns ([0],) for CPU.
    """
    if device is not None:
        d = torch.device(device)
        if d.type == 'cpu':
            return d, [0]
    if not torch.cuda.is_available():
        return torch.device('cpu'), [0]

    total = torch.cuda.device_count()
    usable = []
    for i in range(total):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free >= min_free_mb * 1024 * 1024:
                usable.append(i)
        except Exception:
            usable.append(i)  # can't query; include optimistically

    if not usable:
        usable = [0]  # all GPUs appear full; fall back and hope for the best

    if num_gpus > 0:
        usable = usable[:num_gpus]

    return torch.device(f'cuda:{usable[0]}'), usable


# ======================================================================
# FID
# ======================================================================

def compute_fid(real_images, generated_images, batch_size=64, device=None, num_gpus=0):
    """Compute Frechet Inception Distance between real and generated images.

    Uses torchmetrics FrechetInceptionDistance (torch-fidelity InceptionV3)
    for scores comparable to standard benchmarks.

    Streams batches directly from the numpy arrays to avoid duplicating
    the full dataset as a torch tensor (critical at 1M+ samples).
    Uses DataParallel on the internal InceptionV3 when multiple GPUs
    are available.

    Args:
        real_images: uint8 [0,255] NHWC numpy array.
        generated_images: uint8 [0,255] NHWC numpy array.
        batch_size: Per-GPU batch size for InceptionV3 inference.
        device: torch device string (auto-detected if None).

    Returns:
        float: FID score (lower is better).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    _free_gpu()  # reclaim any cached memory before loading inception
    device, device_ids = _gpu_setup(device, num_gpus)
    effective_batch = batch_size * len(device_ids)

    fid = FrechetInceptionDistance().to(device)
    # Promote only the statistics buffers to float64 for numerical stability.
    # Keeping inception in float32 avoids doubling GPU memory for model weights.
    for name in ('real_features_sum', 'real_features_cov_sum',
                 'real_features_num_samples',
                 'fake_features_sum', 'fake_features_cov_sum',
                 'fake_features_num_samples'):
        if hasattr(fid, name):
            buf = getattr(fid, name)
            setattr(fid, name, buf.to(torch.float64))
    if len(device_ids) > 1:
        fid.inception = torch.nn.DataParallel(fid.inception, device_ids=device_ids)

    def _fid_loop(images, real, desc):
        for i in tqdm(range(0, len(images), effective_batch),
                      desc=desc,
                      total=(len(images) + effective_batch - 1) // effective_batch):
            batch = torch.from_numpy(
                images[i:i + effective_batch]
            ).permute(0, 3, 1, 2).to(device)
            fid.update(batch, real=real)
            del batch
            torch.cuda.empty_cache()

    with torch.no_grad():
        _fid_loop(real_images, real=True, desc='FID (real)')
        _fid_loop(generated_images, real=False, desc='FID (gen)')
        score = fid.compute().item()
    del fid
    _free_gpu()
    return score


# ======================================================================
# Inception Score
# ======================================================================

def compute_inception_score(generated_images, batch_size=64, splits=10, device=None, num_gpus=0):
    """Compute Inception Score for generated images.

    Uses torchmetrics InceptionScore (torch-fidelity InceptionV3)
    for scores comparable to standard benchmarks.

    Args:
        generated_images: uint8 [0,255] NHWC numpy array.
        batch_size: Per-GPU batch size for InceptionV3 inference.
        splits: Number of splits for mean/std computation.
        device: torch device string.

    Returns:
        (mean_is, std_is): Mean and std of IS across splits.
    """
    from torchmetrics.image.inception import InceptionScore

    device, device_ids = _gpu_setup(device, num_gpus)
    effective_batch = batch_size * len(device_ids)

    inception = InceptionScore(splits=splits).to(device)
    if len(device_ids) > 1:
        inception.inception = torch.nn.DataParallel(inception.inception, device_ids=device_ids)

    with torch.no_grad():
        for i in tqdm(range(0, len(generated_images), effective_batch),
                      desc='IS',
                      total=(len(generated_images) + effective_batch - 1) // effective_batch):
            batch = torch.from_numpy(
                generated_images[i:i + effective_batch]
            ).permute(0, 3, 1, 2).to(device)
            inception.update(batch)
            del batch

        mean_is, std_is = inception.compute()
    del inception
    _free_gpu()
    return float(mean_is), float(std_is)


# ======================================================================
# FID per class
# ======================================================================

def compute_fid_per_class(real_images, real_class_labels, generated_images, gen_class_labels,
                           batch_size=64, device=None, num_gpus=0):
    """Compute per-class FID by extracting InceptionV3 features once.

    Extracts 2048-dim InceptionV3 features for all real and generated images
    in a single pass each, then computes FID independently for every class
    using scipy.linalg.sqrtm.  A small eps * I regularisation is added to
    each covariance matrix to handle the rank-deficient case (only ~50
    samples per class in the standard ImageNet val split).

    Args:
        real_images: uint8 [0,255] NHWC numpy array of real images.
        real_class_labels: int numpy array of class indices for real_images.
        generated_images: uint8 [0,255] NHWC numpy array of generated images.
        gen_class_labels: int numpy array of class indices for generated_images.
        batch_size: Per-GPU batch size for InceptionV3 inference.
        device: torch device string (auto-detected if None).
        num_gpus: Number of GPUs to use (0 = all available).

    Returns:
        dict[int, float]: Mapping class_idx -> per-class FID (lower is better).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from scipy import linalg

    _free_gpu()
    device, device_ids = _gpu_setup(device, num_gpus)
    effective_batch = batch_size * len(device_ids)

    fid_metric = FrechetInceptionDistance().to(device)
    if len(device_ids) > 1:
        fid_metric.inception = torch.nn.DataParallel(fid_metric.inception, device_ids=device_ids)

    def _extract(images, desc):
        feats = []
        for i in tqdm(range(0, len(images), effective_batch),
                      desc=desc,
                      total=(len(images) + effective_batch - 1) // effective_batch):
            batch = torch.from_numpy(
                images[i:i + effective_batch]
            ).permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                out = fid_metric.inception(batch)
                f = out['2048'] if isinstance(out, dict) else out
            feats.append(f.cpu().float().numpy())
            del batch
            torch.cuda.empty_cache()
        return np.concatenate(feats, axis=0)

    real_feats = _extract(real_images, 'FID/class (real)')
    gen_feats = _extract(generated_images, 'FID/class (gen)')
    del fid_metric
    _free_gpu()

    real_class_labels = np.asarray(real_class_labels)
    gen_class_labels = np.asarray(gen_class_labels)

    def _fid_from_feats(r, g, eps=1e-6):
        if len(r) < 2 or len(g) < 2:
            return float('nan')
        mu_r, mu_g = r.mean(0), g.mean(0)
        sr = np.cov(r.T) + np.eye(r.shape[1]) * eps
        sg = np.cov(g.T) + np.eye(g.shape[1]) * eps
        diff = mu_r - mu_g
        cov_sqrt, _ = linalg.sqrtm(sr @ sg, disp=False)
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        return float(diff @ diff + np.trace(sr + sg - 2 * cov_sqrt))

    classes = np.unique(gen_class_labels)

    def _compute_one(cls):
        r = real_feats[real_class_labels == cls]
        g = gen_feats[gen_class_labels == cls]
        return int(cls), _fid_from_feats(r, g)

    per_class = {}
    for cls in tqdm(classes, desc='FID/class (compute)', total=len(classes)):
        cls_idx, score = _compute_one(cls)
        per_class[cls_idx] = score
    return per_class


# ======================================================================
# Inception Score per class
# ======================================================================

def compute_inception_score_per_class(generated_images, class_labels,
                                       batch_size=64, device=None, num_gpus=0):
    """Compute per-class Inception Score from a single InceptionV3 pass.

    Extracts InceptionV3 softmax probabilities for all generated images, then
    for each class computes IS = exp(mean KL(p(y|x) || p_y)) where p_y is the
    marginal distribution over images in that class.

    Args:
        generated_images: uint8 [0,255] NHWC numpy array.
        class_labels: int numpy array of class indices for generated_images.
        batch_size: Per-GPU batch size for InceptionV3 inference.
        device: torch device string (auto-detected if None).
        num_gpus: Number of GPUs to use (0 = all available).

    Returns:
        dict[int, float]: Mapping class_idx -> per-class IS (higher is better).
    """
    from torchmetrics.image.inception import InceptionScore

    _free_gpu()
    device, device_ids = _gpu_setup(device, num_gpus)
    effective_batch = batch_size * len(device_ids)

    is_metric = InceptionScore().to(device)
    if len(device_ids) > 1:
        is_metric.inception = torch.nn.DataParallel(is_metric.inception, device_ids=device_ids)

    all_probs = []
    for i in tqdm(range(0, len(generated_images), effective_batch),
                  desc='IS/class',
                  total=(len(generated_images) + effective_batch - 1) // effective_batch):
        batch = torch.from_numpy(
            generated_images[i:i + effective_batch]
        ).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            out = is_metric.inception(batch)
            logits = out['logits_unbiased'] if isinstance(out, dict) else out
            probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
        del batch
        torch.cuda.empty_cache()

    del is_metric
    _free_gpu()
    all_probs = np.concatenate(all_probs, axis=0)  # (N, 1000)
    class_labels = np.asarray(class_labels)

    def _is_from_probs(probs):
        if len(probs) < 2:
            return float('nan')
        p_y = probs.mean(0)
        kl = (probs * (np.log(probs + 1e-10) - np.log(p_y + 1e-10))).sum(1)
        return float(np.exp(kl.mean()))

    per_class = {}
    for cls in np.unique(class_labels):
        mask = class_labels == cls
        per_class[int(cls)] = _is_from_probs(all_probs[mask])
    return per_class


# ======================================================================
# CLIP Score
# ======================================================================

def compute_clip_score(images, class_labels, class_names, batch_size=64, device=None, num_gpus=0,
                        return_per_class=False):
    """Compute CLIP Score between generated images and their class labels.

    Uses openai/clip-vit-base-patch16 for standard-benchmark-compatible
    scores (on the 100x scale). Uses DataParallel for multi-GPU inference.

    Args:
        images: uint8 [0,255] NHWC numpy array.
        class_labels: numpy array of integer class labels (indices into class_names).
        class_names: list of class name strings.
        batch_size: Per-GPU batch size for CLIP inference.
        device: torch device string.
        return_per_class: If True, also return a dict mapping class_idx -> mean score.

    Returns:
        float: Mean CLIP score (higher is better, ~0-100 scale).
        If return_per_class=True, returns (float, dict[int, float]).
    """
    from collections import defaultdict
    from transformers import CLIPModel, CLIPProcessor

    device, device_ids = _gpu_setup(device, num_gpus)
    effective_batch = batch_size * len(device_ids)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval().to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    prompts = [f"a photo of a {class_names[label]}" for label in class_labels]

    weighted_sum = 0.0
    total_count = 0
    all_scores = []
    num_batches = (len(images) + effective_batch - 1) // effective_batch

    for i in tqdm(range(0, len(images), effective_batch), desc='CLIP', total=num_batches):
        batch_imgs = images[i:i + effective_batch]
        batch_prompts = prompts[i:i + effective_batch]
        actual_size = len(batch_imgs)

        # Pad last batch to be divisible by num GPUs — DataParallel gather fails when
        # the batch splits unevenly because CLIP's logits_per_image output is square [n, n].
        n_gpus = len(device_ids)
        if n_gpus > 1 and actual_size % n_gpus != 0:
            pad = n_gpus - (actual_size % n_gpus)
            batch_imgs = list(batch_imgs) + [batch_imgs[-1]] * pad
            batch_prompts = batch_prompts + [batch_prompts[-1]] * pad

        # Feed numpy HWC arrays directly to the processor
        pil_imgs = [batch_imgs[j] for j in range(len(batch_imgs))]
        inputs = processor(text=batch_prompts, images=pil_imgs,
                           return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        img_feats = outputs.image_embeds[:actual_size]
        txt_feats = outputs.text_embeds[:actual_size]
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        scores = (img_feats * txt_feats).sum(dim=-1) * 100.0
        weighted_sum += scores.sum().item()
        total_count += len(batch_imgs)
        all_scores.extend(scores.cpu().tolist())

    del model, processor
    _free_gpu()
    global_mean = float(weighted_sum / total_count)

    if not return_per_class:
        return global_mean

    class_scores = defaultdict(list)
    for idx, score in zip(class_labels, all_scores):
        class_scores[int(idx)].append(score)
    per_class = {cls: float(np.mean(vals)) for cls, vals in class_scores.items()}
    return global_mean, per_class


# ======================================================================
# PickScore
# ======================================================================

def compute_pick_score(images, prompts, batch_size=32, device=None, num_gpus=0,
                        class_labels=None, return_per_class=False):
    """Compute PickScore between generated images and text prompts.

    Uses PickScore_v1 (fine-tuned from CLIP-H on Pick-a-Pic dataset).
    Uses DataParallel for multi-GPU inference via the model's forward()
    which returns both image and text embeddings in a single pass.

    Args:
        images: uint8 [0,255] NHWC numpy array.
        prompts: list of text prompt strings (one per image).
        batch_size: Per-GPU batch size for inference.
        device: torch device string.
        class_labels: Optional numpy array of integer class labels (needed for return_per_class).
        return_per_class: If True, also return a dict mapping class_idx -> mean score.

    Returns:
        float: Mean PickScore (higher is better).
        If return_per_class=True, returns (float, dict[int, float]).
    """
    from collections import defaultdict
    from transformers import AutoProcessor, AutoModel

    device, device_ids = _gpu_setup(device, num_gpus)
    effective_batch = batch_size * len(device_ids)

    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    _base_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    # Access logit_scale before wrapping with DataParallel
    logit_scale = _base_model.logit_scale.exp().item()

    # Wrap to return only (image_embeds, text_embeds) so DataParallel can
    # gather them without a shape mismatch.  The CLIP forward also returns
    # logits_per_image / logits_per_text whose shape is [local_batch,
    # local_batch] — these are different on each GPU when the last mini-batch
    # is not evenly divisible, which causes the RuntimeError on gather.
    class _EmbedOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, pixel_values):
            out = self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            return out.image_embeds, out.text_embeds

    model = _EmbedOnly(_base_model)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    weighted_sum = 0.0
    total_count = 0
    all_scores = []
    num_batches = (len(images) + effective_batch - 1) // effective_batch

    for i in tqdm(range(0, len(images), effective_batch), desc='PickScore', total=num_batches):
        batch_imgs = images[i:i + effective_batch]
        batch_prompts = list(prompts[i:i + effective_batch])

        pil_images = [Image.fromarray(img) for img in batch_imgs]

        image_inputs = processor(
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        text_inputs = processor(
            text=batch_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            image_embs, text_embs = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=image_inputs["pixel_values"],
            )
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            scores = logit_scale * (image_embs * text_embs).sum(dim=-1)
            weighted_sum += scores.sum().item()
            total_count += len(batch_imgs)
            all_scores.extend(scores.cpu().tolist())

    del model, processor
    _free_gpu()
    global_mean = float(weighted_sum / total_count)

    if not return_per_class or class_labels is None:
        return global_mean

    class_scores = defaultdict(list)
    for idx, score in zip(class_labels, all_scores):
        class_scores[int(idx)].append(score)
    per_class = {cls: float(np.mean(vals)) for cls, vals in class_scores.items()}
    return global_mean, per_class


# ======================================================================
# ImageReward
# ======================================================================

def compute_image_reward(images, prompts, batch_size=64, device=None, num_gpus=0):
    """Compute ImageReward scores for generated images and their prompts.

    Uses ImageReward-v1.0 human preference reward model.
    Batch size is scaled by GPU count for throughput (ImageReward's
    score() API handles device placement internally).

    Args:
        images: uint8 [0,255] NHWC numpy array.
        prompts: list of text prompt strings (one per image).
        batch_size: Per-GPU batch size for inference.
        device: torch device string.

    Returns:
        float: Mean ImageReward score (higher is better).
    """
    import ImageReward as reward

    device, device_ids = _gpu_setup(device, num_gpus)
    effective_batch = batch_size * len(device_ids)

    model = reward.load("ImageReward-v1.0", device=device)

    all_scores = []
    num_batches = (len(images) + effective_batch - 1) // effective_batch

    for i in tqdm(range(0, len(images), effective_batch), desc='ImageReward', total=num_batches):
        batch_imgs = [Image.fromarray(img) for img in images[i:i + effective_batch]]
        batch_prompts = list(prompts[i:i + effective_batch])
        with torch.no_grad():
            scores = model.score(batch_prompts, batch_imgs)
        if isinstance(scores, (float, int)):
            all_scores.append(float(scores))
        else:
            all_scores.extend([float(s) for s in scores])

    del model
    _free_gpu()
    return float(np.mean(all_scores))


# ======================================================================
# Save Results
# ======================================================================

def save_metrics(metrics_dict, output_dir, model_name):
    """Save metrics as CSV to {output_dir}/{model_name}/{model_name}_metrics.csv.

    Args:
        metrics_dict: dict of metric_name -> value.
        output_dir: Base output directory.
        model_name: Model identifier string for subfolder/file naming.

    Returns:
        str: Path to saved CSV file.
    """
    save_dir = os.path.join(output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{model_name}_metrics.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics_dict.items():
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])

    return csv_path


def save_per_class_metrics(per_class_metrics, csv_path, class_names):
    """Save per-class metrics to a CSV file.

    Args:
        per_class_metrics: dict[str, dict[int, float]] — metric_name → {class_idx → value}.
        csv_path: Full path to write the CSV (including filename).
        class_names: List of class name strings indexed by class_idx.

    Returns:
        str: Path to the saved CSV file.
    """
    metric_names = list(per_class_metrics.keys())
    num_classes = len(class_names)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_idx', 'class_name'] + metric_names)
        for cls_idx in range(num_classes):
            name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
            row = [cls_idx, name]
            for m in metric_names:
                val = per_class_metrics[m].get(cls_idx, float('nan'))
                row.append(f"{val:.6f}" if not np.isnan(val) else 'nan')
            writer.writerow(row)
    return csv_path


def save_density_plots(per_class_metrics, plot_path, class_names=None):
    """Save KDE density plots of per-class metric distributions.

    One column per metric: a KDE density plot on top and a bar plot on the
    bottom showing the top 10 and bottom 10 classes with a dividing line.

    Args:
        per_class_metrics: dict[str, dict[int, float]] — metric_name → {class_idx → value}.
        plot_path: Full path to save the PNG (including filename).
        class_names: Optional list of class name strings indexed by class_idx.

    Returns:
        str: Path to the saved PNG file.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    metric_names = list(per_class_metrics.keys())
    n = len(metric_names)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9),
                             gridspec_kw={'height_ratios': [3, 2]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, metric in enumerate(metric_names):
        # --- Row 0: KDE density plot ---
        ax = axes[0, i]
        values = np.array(list(per_class_metrics[metric].values()), dtype=float)
        values = values[~np.isnan(values)]
        kde = gaussian_kde(values)
        x = np.linspace(values.min(), values.max(), 300)
        ax.plot(x, kde(x))
        ax.axvline(values.mean(), linestyle='--', label=f'mean={values.mean():.2f}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.set_title(f'Per-class {metric} distribution')
        ax.legend()

        # --- Row 1: Top/Bottom 10 bar plot ---
        bar_ax = axes[1, i]
        class_vals = {k: v for k, v in per_class_metrics[metric].items()
                      if not np.isnan(v)}
        sorted_classes = sorted(class_vals.items(), key=lambda x: x[1])
        bottom10 = sorted_classes[:10]
        top10 = sorted_classes[-10:]
        all20 = bottom10 + top10

        x_pos = np.arange(20)
        bar_vals = [v for _, v in all20]
        bar_labels = [
            (class_names[k] if class_names and k < len(class_names) else str(k))
            for k, _ in all20
        ]
        colors = ['#d73027'] * 10 + ['#1a9850'] * 10

        bar_ax.bar(x_pos, bar_vals, color=colors)
        bar_ax.axvline(x=9.5, color='black', linestyle='-', linewidth=1.5)
        bar_ax.text(4.5 / 19, 0.97, 'Bottom 10', transform=bar_ax.transAxes,
                    ha='center', va='top', fontsize=9, color='#d73027', fontweight='bold')
        bar_ax.text(14.5 / 19, 0.97, 'Top 10', transform=bar_ax.transAxes,
                    ha='center', va='top', fontsize=9, color='#1a9850', fontweight='bold')
        bar_ax.set_xticks(x_pos)
        bar_ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=7)
        bar_ax.set_ylabel(metric)
        bar_ax.set_title(f'Top/Bottom 10 classes — {metric}')

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_sample_grid(images, output_dir, model_name, nrow=5, labels=None, class_names=None):
    """Save sample images as a grid PNG.

    Selects 20 evenly-spaced images from the full generated set.
    If labels and class_names are provided, renders the class name above each image.

    Saves to {output_dir}/{model_name}/{model_name}_sample_generations.png.

    Args:
        images: uint8 [0,255] NHWC numpy array (full generated set).
        output_dir: Base output directory.
        model_name: Model identifier string.
        nrow: Number of images per row in the grid (default: 5).
        labels: Optional int array of class indices, same length as images.
        class_names: Optional list of class name strings indexed by label.

    Returns:
        str: Path to saved PNG file.
    """
    from torchvision.utils import make_grid
    from PIL import ImageDraw

    # Select 20 evenly-spaced images across the full set
    indices = np.linspace(0, len(images) - 1, 20, dtype=int).tolist()
    selected = np.asarray(images[indices])  # (20, H, W, 3) uint8

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

    save_dir = os.path.join(output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f"{model_name}_sample_generations.png")

    # NHWC uint8 numpy -> NCHW float tensor [0, 1]
    img_tensors = torch.from_numpy(selected).permute(0, 3, 1, 2).float() / 255.0
    grid = make_grid(img_tensors, nrow=nrow, padding=2)

    # Save as PIL image
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(png_path)

    return png_path
