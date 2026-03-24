import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from absl import logging
from jax.experimental import multihost_utils
from PIL import Image
from tqdm import tqdm

from .jax_fid import inception, resize
from .jax_fid.fid import compute_frechet_distance
from .logging_util import log_for_0

compute_fid = compute_frechet_distance


def build_jax_inception(batch_size=200):
    # jax model
    logging.info("Initializing InceptionV3")
    model = inception.InceptionV3(pretrained=True)
    inception_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 16, 16, 3)))
    logging.info("Initialized InceptionV3")
    inception_fn = jax.jit(functools.partial(model.apply, train=False))

    fake_x = np.zeros((batch_size, 299, 299, 3), dtype=np.float32)
    lowered = inception_fn.lower(inception_params, jax.lax.stop_gradient(fake_x))
    logging.info('Start compiling inception_fn...')
    t_start = time.time()
    compiled = lowered.compile()
    logging.info(f'End compiling: {(time.time() - t_start):.4f} seconds.')
    inception_fn = compiled

    inception_net = {"params": inception_params, "fn": inception_fn}
    return inception_net


def get_reference(cache_path, inception_net=None, batch_size=200, num_samples=50000):
    # Load ref_mu and ref_sigma from npz file
    assert os.path.exists(cache_path), f"Cache file must exist: {cache_path}"
    
    logging.info(f"Loading ref_mu and ref_sigma from {cache_path}")
    os.system('md5sum ' + cache_path)
    with np.load(cache_path) as data:
        if "ref_mu" in data:
            ref_mu, ref_sigma = data["ref_mu"], data["ref_sigma"]
        else:
            raise NotImplementedError

    ref = {"mu": ref_mu, "sigma": ref_sigma}
    return ref


def compute_stats(
    samples,
    inception_net,
    batch_size=200,
    num_workers=12,
    fid_samples=50000,
    mode = "legacy_tensorflow"
):
    inception_fn = inception_net["fn"]
    inception_params = inception_net["params"]

    num_samples = len(samples)
    pad = int(np.ceil(num_samples / batch_size)) * batch_size - num_samples
    samples = np.concatenate([samples, np.zeros((pad, *samples.shape[1:]), dtype=np.uint8)])
    assert len(samples) % batch_size == 0

    dataset = ResizeDataset(samples, mode=mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    l_feats = []
    for i, x in enumerate(dataloader):
        if i % 50 == 0:
            logging.info(f"Evaluating {i} / {len(dataloader)}: {list(x.shape)}")
        x = resize.forward(x)  # match the Pytorch version
        x = x.numpy().transpose(0,2,3,1)
        pred = inception_fn(inception_params, jax.lax.stop_gradient(x))
        pred = pred.squeeze(axis=1).squeeze(axis=1)
        l_feats.append(pred)
    np_feats = jnp.concatenate(l_feats)
    np_feats = np_feats[:num_samples]
    
    all_feats = multihost_utils.process_allgather(np_feats)
    all_feats = all_feats.reshape(-1, np_feats.shape[-1])
    all_feats = jax.device_get(all_feats)

    all_feats = all_feats[:fid_samples]
    mu = np.mean(all_feats, axis=0)
    sigma = np.cov(all_feats, rowvar=False)

    return mu, sigma


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_np = self.files[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


# this is from the cleanfid package
def build_resizer(mode):
    if mode == "clean":
        return make_resizer("PIL", False, "bicubic", (299,299))
    # if using legacy tensorflow, do not manually resize outside the network
    elif mode == "legacy_tensorflow":
        return lambda x: x
    elif mode == "legacy_pytorch":
        return make_resizer("PyTorch", False, "bilinear", (299, 299))
    else:
        raise ValueError(f"Invalid mode {mode} specified")


"""
Construct a function that resizes a numpy image based on the
flags passed in.
"""
def make_resizer(library, quantize_after, filter, output_size):
    if library == "PIL" and quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX
        }
        def func(x):
            x = Image.fromarray(x)
            x = x.resize(output_size, resample=name_to_filter[filter])
            x = np.asarray(x).clip(0, 255).astype(np.uint8)
            return x
    elif library == "PIL" and not quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX
        }
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=name_to_filter[filter])
            return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    elif library == "TensorFlow":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        import tensorflow as tf
        def func(x):
            x = tf.constant(x)[tf.newaxis, ...]
            x = tf.image.resize(x, output_size, method=filter)
            x = x[0, ...].numpy().clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    else:
        raise NotImplementedError('library [%s] is not included' % library)
    return func


def compute_fid_stats(imagenet_root, output_dir, image_size, batch_size=200, overwrite=False):
    """Compute and save FID statistics for ImageNet using distributed loading and chunked gathering."""
    from utils.data_util import create_imagenet_dataloader
    
    log_for_0("Starting FID statistics computation...")
    
    # Output path for FID stats
    fid_stats_path = os.path.join(output_dir, f'imagenet_{image_size}_fid_stats.npz')
    
    # Check if already exists
    if not overwrite and os.path.exists(fid_stats_path):
        log_for_0(f"FID stats already exist at {fid_stats_path}, skipping...")
        return fid_stats_path
    
    # Build Inception model
    inception_net = build_jax_inception(batch_size=batch_size)
    inception_fn = inception_net["fn"]
    inception_params = inception_net["params"]
    
    # Create dataloader for training set (for FID reference)
    # Use num_workers=0 to avoid fork() incompatibility with JAX multithreading
    dataloader, dataset_size, true_total_samples = create_imagenet_dataloader(
        imagenet_root, 'train', batch_size, image_size, num_workers=0, for_fid=True
    )
    
    log_for_0(f"Computing FID features for {dataset_size} samples per worker...")
    log_for_0(f"Expected batches per worker: {len(dataloader)}")
    
    # Process data batch by batch and accumulate features
    log_for_0("Processing batches and computing features...")
    all_features_list = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        images, labels = batch
        
        # Convert images to numpy array format
        if isinstance(images, list):
            images_np = np.stack(images, axis=0)
        else:
            images_np = np.array(images)
        
        # Compute features for this batch directly
        batch_features = compute_batch_features(
            images_np, inception_fn, inception_params, batch_size
        )
        
        # Move to CPU and accumulate
        batch_features_cpu = jax.device_get(batch_features)
        all_features_list.append(batch_features_cpu)
        
        if batch_idx % 100 == 0:
            log_for_0(f"Worker {jax.process_index()}: Processed {batch_idx}/{len(dataloader)} batches")
    
    # Concatenate all local features from this worker
    local_features = np.concatenate(all_features_list, axis=0)
    log_for_0(f"Worker {jax.process_index()}: Local features shape: {local_features.shape}")
    
    # Clear feature list to free memory
    del all_features_list
    
    # Gather features across all workers using chunked approach to avoid OOM
    log_for_0("Gathering features across workers using chunked approach...")
    
    # Use smaller chunk size to avoid OOM (10K samples per chunk)
    chunk_size = 10000
    all_gathered_features = []
    
    for chunk_start in range(0, local_features.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, local_features.shape[0])
        local_chunk = local_features[chunk_start:chunk_end]
        
        log_for_0(f"Worker {jax.process_index()}: Gathering chunk {chunk_start//chunk_size + 1}, "
                 f"samples {chunk_start}:{chunk_end} ({local_chunk.shape[0]} samples)")
        
        # Convert to JAX array and gather this chunk across all processes
        local_chunk_jax = jnp.array(local_chunk)
        
        # Gather this chunk from all workers
        gathered_chunk = multihost_utils.process_allgather(local_chunk_jax)
        gathered_chunk = gathered_chunk.reshape(-1, gathered_chunk.shape[-1])
        
        # Move to CPU to free memory
        gathered_chunk_cpu = jax.device_get(gathered_chunk)
        all_gathered_features.append(gathered_chunk_cpu)
        
        log_for_0(f"Worker {jax.process_index()}: Successfully gathered chunk {chunk_start//chunk_size + 1}, "
                    f"total shape: {gathered_chunk_cpu.shape}")
            
    # Concatenate all gathered chunks
    all_features_gathered = np.concatenate(all_gathered_features, axis=0)
    log_for_0(f"Total features shape before truncation: {all_features_gathered.shape}")
    
    # Truncate the padding by gathering
    if all_features_gathered.shape[0] != true_total_samples:
        log_for_0("Truncating to expected number of samples to fix padding...")
        all_features_gathered = all_features_gathered[:true_total_samples]
    
    log_for_0(f"Final features shape after truncation: {all_features_gathered.shape}")
    
    # Clear local features to free memory
    del local_features
    
    # Compute statistics
    log_for_0("Computing final statistics...")
    mu = np.mean(all_features_gathered, axis=0)
    sigma = np.cov(all_features_gathered, rowvar=False)
    
    # Save statistics
    os.makedirs(os.path.dirname(fid_stats_path), exist_ok=True)
    np.savez(fid_stats_path, ref_mu=mu, ref_sigma=sigma)
    log_for_0(f"FID statistics saved to {fid_stats_path}")
    
    return fid_stats_path


def compute_batch_features(batch_images, inception_fn, inception_params, batch_size):
    """Compute Inception features for a batch of images."""
    actual_batch_size = batch_images.shape[0]
    
    # Convert uint8 [0,255] numpy to float32 [0,255] tensor
    x = torch.tensor(batch_images, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2)  # BHWC → BCHW for PyTorch
    
    # Apply resize and normalization, then convert to JAX format
    x = resize.forward(x)  # Resize to 299x299 and normalize to [-1,1]
    x = x.numpy().transpose(0, 2, 3, 1)  # BCHW → BHWC for JAX
    
    # Pad batch to expected size if needed (for JAX compilation compatibility)
    if actual_batch_size < batch_size:
        # Pad with zeros to reach expected batch size
        padding_size = batch_size - actual_batch_size
        padding_shape = (padding_size,) + x.shape[1:]
        padding = np.zeros(padding_shape, dtype=x.dtype)
        x_padded = np.concatenate([x, padding], axis=0)
    else:
        x_padded = x
    
    # Extract Inception features
    pred = inception_fn(inception_params, jax.lax.stop_gradient(x_padded))
    pred = pred.squeeze(axis=1).squeeze(axis=1)
    
    # Return only the features for actual samples (remove padding)
    pred = pred[:actual_batch_size]
    
    return jax.device_get(pred)