"""Fine-tune MeanFlow on Mini-ImageNet with class conditioning.

This script loads a pretrained MeanFlow checkpoint (trained on ImageNet)
and fine-tunes it on Mini-ImageNet with full weight updates, WandB logging,
and multi-GPU (pmap) data parallelism.

Usage:
    python finetune_miniimagenet.py \
        --data_path=/path/to/mini_imagenet \
        --pretrained_ckpt=/path/to/checkpoint \
        --workdir=/path/to/output
"""
import csv
import os
import random as pyrandom
import warnings
warnings.filterwarnings("ignore", message=".*Flax classes are deprecated.*")
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import jax

if os.environ.get("COORDINATOR_ADDRESS") is not None:
    jax.distributed.initialize()

import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import torch
import torch.utils.data
from absl import app, flags, logging
from flax import jax_utils
from flax import serialization
from flax.training import checkpoints, common_utils, train_state
from jax import lax, random
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import wandb
from meanflow import MeanFlow, generate
from utils.ckpt_util import restore_checkpoint, save_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.fid_util import build_jax_inception, compute_fid, compute_stats
from utils.info_util import print_params
from utils.input_pipeline import center_crop_arr, worker_init_fn
from utils.logging_util import Timer, log_for_0, supress_checkpt_info
from utils.sample_util import generate_fid_samples
from utils.vae_util import LatentManager
from utils.vis_util import make_grid_visualization

supress_checkpt_info()
warnings.filterwarnings("ignore")

# ======================================================================
# CLI Flags
# ======================================================================

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', None, 'Path to Mini-ImageNet root (CSV + images/ folder, or folder-based)')
flags.DEFINE_string('pretrained_ckpt', None, 'Path to pretrained MeanFlow checkpoint directory')
flags.DEFINE_string('workdir', 'outputs_finetune', 'Directory to save fine-tuned checkpoints and logs')
flags.DEFINE_string('model_cls', 'DiT_B_4', 'DiT model variant')
flags.DEFINE_integer('batch_size', 64, 'Total batch size across all GPUs')
flags.DEFINE_integer('num_epochs', 100, 'Number of fine-tuning epochs')
flags.DEFINE_float('learning_rate', 5e-5, 'Learning rate for fine-tuning')
flags.DEFINE_string('vae_type', 'mse', 'VAE variant (mse or ema)')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_string('wandb_project', 'meanflow_reimplementation', 'WandB project name')
flags.DEFINE_string('wandb_run_name', None, 'WandB run name (auto-generated if not set)')
flags.DEFINE_integer('log_per_step', 10, 'Log metrics every N steps')
flags.DEFINE_integer('sample_per_epoch', 50, 'Generate sample images every N epochs')
flags.DEFINE_integer('checkpoint_per_epoch', 10, 'Save checkpoint every N epochs')
flags.DEFINE_integer('fid_per_epoch', 200, 'Compute FID every N epochs')

# ======================================================================
# Dataset
# ======================================================================

class MiniImageNetDataset(torch.utils.data.Dataset):
    """
    Mini-ImageNet dataset loader.

    Supports two formats:
      1. CSV + flat images:
           data_path/train.csv  (columns: filename, label)
           data_path/images/*.jpg

      2. Folder-based (ImageFolder):
           data_path/train/n01532829/*.jpg
    """

    def __init__(self, data_path, split='train', transform=None):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.samples = []
        self.classes = []

        split_dir = self.data_path / split
        if split_dir.exists() and split_dir.is_dir():
            self._load_from_folders(split_dir)
        elif (self.data_path / f"{split}.csv").exists():
            self._load_from_csv()
        else:
            raise ValueError(
                f"No data found at {data_path} for split '{split}'. "
                f"Expected either {split_dir}/ directory or {split}.csv file."
            )

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_path} for split '{split}'")

        log_for_0(f'MiniImageNetDataset: {len(self.samples)} images, {len(self.classes)} classes (split={split})')

    def _load_from_folders(self, split_dir):
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        for class_idx, class_dir in enumerate(class_dirs):
            for ext in ['*.jpg', '*.JPEG', '*.png']:
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((img_path, class_idx))

    def _load_from_csv(self):
        csv_path = self.data_path / f"{self.split}.csv"
        images_dir = self.data_path / "images"
        if not images_dir.exists():
            images_dir = self.data_path

        class_set = set()
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get('filename', row.get('image', ''))
                label = row.get('label', row.get('class', ''))
                img_path = images_dir / img_name
                if img_path.exists():
                    rows.append((img_path, label))
                    class_set.add(label)

        self.classes = sorted(class_set)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for img_path, label in rows:
            self.samples.append((img_path, class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx


def create_miniimagenet_dataloader(data_path, batch_size, image_size=256):
    """Create a PyTorch DataLoader for Mini-ImageNet with appropriate transforms."""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    ds = MiniImageNetDataset(data_path, split='train', transform=transform)

    rank = jax.process_index()
    sampler = DistributedSampler(
        ds,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=True,
    )
    steps_per_epoch = len(loader)
    return loader, steps_per_epoch


# ======================================================================
# Batch Preparation & VAE Encoding
# ======================================================================

def prepare_raw_image_batch(batch):
    """Reformat a raw-image batch from PyTorch DataLoader for pmap.

    Args:
        batch: (images [B, 3, H, W], labels [B]) from DataLoader

    Returns:
        dict with 'image' (local_devices, device_bs, H, W, 3) numpy
                   'label' (local_devices, device_bs) numpy
    """
    image, label = batch
    local_device_count = jax.local_device_count()

    image = image.permute(0, 2, 3, 1)  # NCHW -> NHWC
    image = image.reshape((local_device_count, -1) + image.shape[1:])
    label = label.reshape(local_device_count, -1)

    return {
        'image': image.numpy(),
        'label': label.numpy(),
    }


def encode_batch_to_latent(images_np, latent_manager, rng):
    """Encode a pmap-shaped batch of raw images to VAE latents.

    Args:
        images_np: (local_devices, device_bs, H, W, 3) numpy, values in [-1, 1]
        latent_manager: LatentManager instance
        rng: JAX PRNG key

    Returns:
        (local_devices, device_bs, latent_H, latent_W, 4) JAX array
    """
    shape = images_np.shape
    num_devices, device_bs = shape[0], shape[1]

    # Flatten devices dimension for encoding
    flat = images_np.reshape((-1,) + shape[2:])  # (B, H, W, 3)
    flat_jax = jnp.array(flat)

    # FlaxAutoencoderKL.encode expects NCHW input, returns NHWC latents
    flat_nchw = jnp.transpose(flat_jax, (0, 3, 1, 2))  # (B, 3, H, W)
    latents = latent_manager.encode(flat_nchw, rng)  # (B, latent_H, latent_W, 4)

    # Reshape for pmap
    latents = latents.reshape((num_devices, device_bs) + latents.shape[1:])
    return latents


# ======================================================================
# Training State & Step
# ======================================================================

class TrainState(train_state.TrainState):
    ema_params: Any


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 4)
    x = jnp.ones(input_shape)
    t = jnp.ones((1,), dtype=int)
    y = jnp.ones((1,), dtype=int)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0('Initializing params...')
    variables = init({'params': key}, x, t, y)
    log_for_0('Initializing params done.')

    param_count = sum(x.size for x in jax.tree.leaves(variables['params']))
    log_for_0("Total trainable parameters: " + str(param_count))
    return variables, variables['params']


def create_train_state(rng, config, model, image_size, lr_value):
    rng, rng_init = random.split(rng)

    _, params = initialized(rng_init, image_size, model)
    ema_params = deepcopy(params)
    ema_params = update_ema(ema_params, params, 0)
    print_params(params['net'])

    tx = optax.adamw(
        learning_rate=lr_value,
        weight_decay=0,
        b2=config.training.adam_b2,
    )
    state = TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        tx=tx,
    )
    return state


def compute_metrics(dict_losses):
    metrics = dict_losses.copy()
    metrics = lax.all_gather(metrics, axis_name='batch')
    metrics = jax.tree.map(lambda x: x.flatten(), metrics)
    return metrics


def train_step_finetune(state, batch, rng_init, config, lr, ema_fn):
    """pmap'd training step. Receives already-encoded latents."""
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))

    images = batch['image']   # (device_bs, latent_H, latent_W, 4)
    labels = batch['label']   # (device_bs,)

    def loss_fn(params):
        variables = {"params": params}
        outputs = state.apply_fn(
            variables,
            imgs=images,
            labels=labels,
            rngs=dict(gen=rng_base),
        )
        loss, dict_losses = outputs
        return loss, (dict_losses,)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')

    dict_losses, = aux[1]
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr

    new_state = state.apply_gradients(grads=grads)

    ema_value = ema_fn(state.step)
    new_ema = update_ema(new_state.ema_params, new_state.params, ema_value)
    new_state = new_state.replace(ema_params=new_ema)

    return new_state, metrics


# ======================================================================
# Sampling
# ======================================================================

def sample_step(variable, sample_idx, model, rng_init, device_batch_size, config):
    rng_sample = random.fold_in(rng_init, sample_idx)
    images = generate(variable, model, rng_sample, n_sample=device_batch_size, config=config)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    return images


def run_p_sample_step(p_sample_step, state, sample_idx, latent_manager, ema=True):
    variable = {"params": state.ema_params if ema else state.params}
    latent = p_sample_step(variable, sample_idx=sample_idx)
    latent = latent.reshape(-1, *latent.shape[2:])

    samples = latent_manager.decode(latent)
    assert not jnp.any(jnp.isnan(samples)), "NaN in decoded samples!"

    samples = samples.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    samples = 127.5 * samples + 128.0
    samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

    jax.random.normal(random.key(0), ()).block_until_ready()
    return samples


# ======================================================================
# FID Evaluation
# ======================================================================

def compute_reference_stats_from_dataset(data_path, inception_net, image_size=256, batch_size=200):
    """Compute FID reference statistics from the Mini-ImageNet training set."""
    log_for_0('Computing FID reference statistics from training data...')

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),
    ])
    ds = MiniImageNetDataset(data_path, split='train', transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Collect all images as uint8 numpy (NHWC)
    all_images = []
    for images, _ in loader:
        # images: (B, 3, H, W) float [0,1] -> uint8 [0,255] NHWC
        imgs_np = (images.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
        all_images.append(imgs_np)
    all_images = np.concatenate(all_images, axis=0)
    log_for_0(f'Reference images: {all_images.shape[0]} samples')

    mu, sigma = compute_stats(all_images, inception_net, batch_size=batch_size, fid_samples=all_images.shape[0])
    log_for_0('FID reference statistics computed.')
    return {"mu": mu, "sigma": sigma}


def evaluate_fid(state, config, p_sample_step, latent_manager, ref_stats, inception_net, label=""):
    """Generate samples and compute FID against reference stats."""
    num_samples = config.fid.num_samples
    log_for_0(f'[FID {label}] Generating {num_samples} samples...')

    run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager)
    samples_all = generate_fid_samples(
        state, "", config, p_sample_step, run_p_sample_step_inner
    )

    log_for_0(f'[FID {label}] Computing statistics on {samples_all.shape[0]} samples...')
    mu, sigma = compute_stats(samples_all, inception_net, fid_samples=num_samples)
    fid_score = compute_fid(mu, ref_stats["mu"], sigma, ref_stats["sigma"])
    log_for_0(f'[FID {label}] FID = {fid_score:.4f} ({num_samples} samples)')
    return fid_score


# ======================================================================
# Config
# ======================================================================

def get_finetune_config(
    data_path,
    pretrained_ckpt,
    batch_size=64,
    num_epochs=100,
    learning_rate=5e-5,
    model_cls='DiT_B_4',
    vae_type='mse',
    seed=42,
    log_per_step=10,
    sample_per_epoch=50,
    checkpoint_per_epoch=10,
    fid_per_epoch=200,
):
    config = ml_collections.ConfigDict()

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.name = 'miniimagenet'
    dataset.root = data_path
    dataset.num_workers = 4
    dataset.prefetch_factor = 2
    dataset.pin_memory = False
    dataset.cache = False
    dataset.image_size = 32       # latent space (256 // 8)
    dataset.raw_image_size = 256  # input to VAE
    dataset.image_channels = 4
    dataset.num_classes = 1000    # keep pretrained embedding size
    dataset.vae = vae_type

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.learning_rate = learning_rate
    training.batch_size = batch_size
    training.num_epochs = num_epochs
    training.log_per_step = log_per_step
    training.sample_per_epoch = sample_per_epoch
    training.checkpoint_per_epoch = checkpoint_per_epoch
    training.fid_per_epoch = fid_per_epoch
    training.half_precision = False
    training.seed = seed
    training.adam_b2 = 0.95
    training.ema_val = 0.9999
    training.sample_on_training = True

    # MeanFlow method
    config.method = method = ml_collections.ConfigDict()
    method.noise_dist = 'logit_normal'
    method.P_mean = -0.4
    method.P_std = 1.0
    method.data_proportion = 0.75
    method.class_dropout_prob = 0.1
    method.guidance_eq = 'cfg'
    method.omega = 1.0
    method.kappa = 0.5
    method.t_start = 0.0
    method.t_end = 1.0
    method.norm_p = 1.0
    method.norm_eps = 0.01

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.cls = model_cls

    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.seed = 0
    sampling.num_steps = 1
    sampling.schedule = 'default'
    sampling.sampling_timesteps = None
    sampling.num_classes = dataset.num_classes

    # FID
    config.fid = fid = ml_collections.ConfigDict()
    fid.on_training = True
    fid.num_samples = 10000
    fid.device_batch_size = 20
    fid.cache_ref = ''

    config.load_from = pretrained_ckpt
    config.eval_only = False

    return config


# ======================================================================
# Main Training Loop
# ======================================================================

def finetune(config, workdir):
    """Fine-tune MeanFlow on Mini-ImageNet."""
    rng = random.key(config.training.seed)
    image_size = config.dataset.image_size

    # ---- WandB ----
    if jax.process_index() == 0:
        run_name = FLAGS.wandb_run_name or f"finetune_miniimagenet_{config.model.cls}"
        wandb.init(
            project=FLAGS.wandb_project,
            name=run_name,
            config=config.to_dict(),
        )

    # ---- Batch size validation ----
    log_for_0('config.training.batch_size: {}'.format(config.training.batch_size))

    if config.training.batch_size % jax.process_count() > 0:
        raise ValueError('Batch size must be divisible by the number of processes')
    local_batch_size = config.training.batch_size // jax.process_count()
    log_for_0('local_batch_size: {}'.format(local_batch_size))
    log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError('Local batch size must be divisible by the number of local devices')

    # ---- DataLoader ----
    train_loader, steps_per_epoch = create_miniimagenet_dataloader(
        config.dataset.root,
        local_batch_size,
        image_size=config.dataset.raw_image_size,
    )
    log_for_0('Steps per Epoch: {}'.format(steps_per_epoch))

    # ---- Model ----
    model_config = config.model.to_dict()
    model_str = model_config.pop('cls')

    model = MeanFlow(
        model_str=model_str,
        model_config=model_config,
        **config.sampling,
        **config.method,
    )

    # ---- Train State ----
    base_lr = config.training.learning_rate
    state = create_train_state(rng, config, model, image_size, lr_value=base_lr)

    if config.load_from is not None:
        # Restore only params from pretrained checkpoint (skip optimizer state
        # to avoid version mismatch issues with optax/flax).
        # Use orbax to properly load arrays from multiprocess checkpoint.
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt_path = str(Path(config.load_from).resolve())
        raw_ckpt = checkpointer.restore(ckpt_path)
        pretrained_params = raw_ckpt['params']
        pretrained_ema = raw_ckpt.get('ema_params', pretrained_params)
        # Rebuild state with pretrained params and fresh optimizer
        tx = optax.adamw(
            learning_rate=base_lr,
            weight_decay=0,
            b2=config.training.adam_b2,
        )
        state = TrainState.create(
            apply_fn=partial(model.apply, method=model.forward),
            params=pretrained_params,
            ema_params=pretrained_ema,
            tx=tx,
        )
        log_for_0('=' * 60)
        log_for_0('Pretrained weights have been loaded (params only, fresh optimizer).')
        log_for_0(f'  Checkpoint: {config.load_from}')
        log_for_0('=' * 60)

    step_offset = 0
    epoch_offset = 0

    state = jax_utils.replicate(state)
    ema_fn = ema_schedules(config)

    # ---- VAE ----
    latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size, image_size)

    # ---- pmap training step ----
    p_train_step = jax.pmap(
        partial(
            train_step_finetune,
            rng_init=rng,
            config=config,
            lr=base_lr,
            ema_fn=ema_fn,
        ),
        axis_name='batch',
    )

    # ---- pmap sampling step ----
    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(config.sampling.seed),
            device_batch_size=config.fid.device_batch_size,
            config=config,
        ),
        axis_name='batch',
    )

    vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())

    # ---- FID Setup & Pre-Finetuning Evaluation ----
    inception_net = build_jax_inception()
    ref_stats = compute_reference_stats_from_dataset(
        config.dataset.root, inception_net, image_size=config.dataset.raw_image_size
    )

    fid_before = evaluate_fid(
        state, config, p_sample_step, latent_manager, ref_stats, inception_net,
        label="BEFORE fine-tuning",
    )
    log_for_0(f'FID BEFORE fine-tuning: {fid_before:.4f}')
    if jax.process_index() == 0:
        wandb.log({'fid_before_finetune': fid_before}, step=0)

    train_metrics = []
    log_for_0('Initial compilation, this might take some minutes...')

    # ---- Training Loop ----
    for epoch in range(epoch_offset, config.training.num_epochs):

        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0('epoch {}...'.format(epoch))

        # ---- Sample visualization (20 images, 5x4 grid) ----
        if (epoch + 1) % config.training.sample_per_epoch == 0 and config.training.get('sample_on_training', True):
            log_for_0(f'Generating samples at epoch {epoch}...')
            vis_sample = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager)
            vis_sample = vis_sample[:20]
            vis_grid = make_grid_visualization(vis_sample, grid=5)
            if jax.process_index() == 0:
                wandb.log({'samples': wandb.Image(np.array(vis_grid[0]))}, step=epoch * steps_per_epoch)

        # ---- Train ----
        timer = Timer()
        timer.reset()

        for n_batch, batch in enumerate(train_loader):
            step = epoch * steps_per_epoch + n_batch

            # Prepare raw images and encode to latent
            raw_batch = prepare_raw_image_batch(batch)

            rng, rng_vae = random.split(rng)
            latent_images = encode_batch_to_latent(
                raw_batch['image'], latent_manager, rng_vae
            )

            pmap_batch = {
                'image': latent_images,
                'label': raw_batch['label'],
            }

            state, metrics = p_train_step(state, pmap_batch)

            if epoch == epoch_offset and n_batch == 0:
                log_for_0('Initial compilation completed. Reset timer.')
                compilation_time = timer.elapse_with_reset()
                log_for_0('p_train_step compiled in {:.2f}s'.format(compilation_time))

            # ---- Metrics ----
            train_metrics.append(metrics)
            if (step + 1) % config.training.log_per_step == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = jax.tree_util.tree_map(lambda x: float(x.mean()), train_metrics)
                summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
                summary['ep'] = epoch

                log_for_0(
                    'train epoch: %d, step: %d, loss: %.6f, steps/sec: %.2f',
                    epoch, step, summary['loss'], summary['steps_per_second'],
                )

                if jax.process_index() == 0:
                    wandb.log(summary, step=step + 1)

                train_metrics = []

        # ---- Checkpoint ----
        if (
            (epoch + 1) % config.training.checkpoint_per_epoch == 0
            or (epoch + 1) == config.training.num_epochs
        ):
            save_checkpoint(state, workdir)

        # ---- Periodic FID evaluation ----
        if (
            (epoch + 1) % config.training.fid_per_epoch == 0
            and config.fid.on_training
        ):
            fid_score = evaluate_fid(
                state, config, p_sample_step, latent_manager, ref_stats, inception_net,
                label=f"epoch {epoch + 1}",
            )
            log_for_0(f'FID at epoch {epoch + 1}: {fid_score:.4f}')
            if jax.process_index() == 0:
                wandb.log({'fid': fid_score}, step=(epoch + 1) * steps_per_epoch)

    # ---- FID After Fine-Tuning ----
    fid_after = evaluate_fid(
        state, config, p_sample_step, latent_manager, ref_stats, inception_net,
        label="AFTER fine-tuning",
    )
    log_for_0(f'FID AFTER fine-tuning: {fid_after:.4f}')
    if jax.process_index() == 0:
        wandb.log({'fid_after_finetune': fid_after}, step=config.training.num_epochs * steps_per_epoch)

    log_for_0('=' * 60)
    log_for_0(f'FID BEFORE fine-tuning: {fid_before:.4f}')
    log_for_0(f'FID AFTER  fine-tuning: {fid_after:.4f}')
    log_for_0(f'FID improvement: {fid_before - fid_after:.4f}')
    log_for_0('=' * 60)

    # ---- Save final model ----
    save_checkpoint(state, workdir)
    log_for_0(f'Final fine-tuned model saved to {workdir}')

    # ---- Cleanup ----
    jax.random.normal(jax.random.key(0), ()).block_until_ready()

    if jax.process_index() == 0:
        wandb.finish()

    return state


# ======================================================================
# Entry Point
# ======================================================================

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    log_for_0('JAX process: %d / %d', jax.process_index(), jax.process_count())
    log_for_0('JAX local devices: %r', jax.local_devices())

    config = get_finetune_config(
        data_path=FLAGS.data_path,
        pretrained_ckpt=FLAGS.pretrained_ckpt,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.learning_rate,
        model_cls=FLAGS.model_cls,
        vae_type=FLAGS.vae_type,
        seed=FLAGS.seed,
        log_per_step=FLAGS.log_per_step,
        sample_per_epoch=FLAGS.sample_per_epoch,
        checkpoint_per_epoch=FLAGS.checkpoint_per_epoch,
        fid_per_epoch=FLAGS.fid_per_epoch,
    )

    log_for_0('Config:\n{}'.format(config))

    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)
    log_for_0(f'Output directory: {workdir}')

    finetune(config, workdir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['data_path', 'pretrained_ckpt'])
    app.run(main)
