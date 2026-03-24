# from absl import logging
import os

import jax
import jax.numpy as jnp
import numpy as np

from utils.logging_util import log_for_0


def generate_fid_samples(state, workdir, config, p_sample_step, run_p_sample_step, ema=True):
  num_steps = np.ceil(config.fid.num_samples / config.fid.device_batch_size / jax.device_count()).astype(int)
  
  output_dir = os.path.join(workdir, 'samples')
  os.makedirs(output_dir, exist_ok=True)
  
  samples_all = []

  log_for_0('Note: the first sample may be significant slower')
  for step in range(num_steps):
    sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())
    sample_idx = jax.device_count() * step + sample_idx
    log_for_0(f'Sampling step {step} / {num_steps}...')
    samples = run_p_sample_step(p_sample_step, state, sample_idx=sample_idx, ema=ema)
    samples = jax.device_get(samples)
    samples_all.append(samples)

  samples_all = np.concatenate(samples_all, axis=0) 
  # samples_all = samples_all[:config.fid.num_samples]
  return samples_all

