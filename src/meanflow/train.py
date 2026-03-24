"""ImageNet DiT example.

This script trains a DiT on the ImageNet dataset.
The data is loaded using pytorch dataset.
"""
from copy import deepcopy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from clu import metric_writers
from flax import jax_utils
from flax.training import common_utils, train_state
from jax import lax, random
from optax._src.alias import *

import utils.input_pipeline as input_pipeline
from meanflow import MeanFlow, generate
from utils import fid_util, sample_util
from utils.ckpt_util import restore_checkpoint, save_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.info_util import print_params
from utils.logging_util import Timer, log_for_0
from utils.vae_util import LatentManager
from utils.vis_util import make_grid_visualization

#######################################################
# Initialize
#######################################################

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


class TrainState(train_state.TrainState):
  ema_params: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, lr_value
):
  """
  Create initial training state.
  ---
  apply_fn: output a dict, with key 'loss', 'mse'
  """

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

#######################################################
# Train Step
#######################################################

def compute_metrics(dict_losses):
  metrics = dict_losses.copy()
  metrics = lax.all_gather(metrics, axis_name='batch')
  metrics = jax.tree.map(lambda x: x.flatten(), metrics)  # (batch_size,)
  return metrics


def train_step_with_vae(state, batch, rng_init, config, lr, ema_fn, latent_mnger):
  """
  Perform a single training step.
  """
  rng_step = random.fold_in(rng_init, state.step)
  rng_base = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))

  cached = batch['image'] # [B, H, W, C]
  rng_base, rng_vae = random.split(rng_base)
  images = latent_mnger.cached_encode(cached, rng_vae) # [B, H, W, C] sample latent

  labels = batch['label']

  def loss_fn(params):
    """loss function used for training."""
    variables = {
        "params": params,
    }
    outputs = state.apply_fn(
      variables,
      imgs=images,
      labels=labels,
      rngs=dict(gen=rng_base,),
    )
    loss, dict_losses = outputs
    return loss, (dict_losses,)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  grads = lax.pmean(grads, axis_name='batch')

  dict_losses, = aux[1]
  metrics = compute_metrics(dict_losses)
  metrics["lr"] = lr

  new_state = state.apply_gradients(
    grads=grads,
  )

  ema_value = ema_fn(state.step)
  new_ema = update_ema(new_state.ema_params, new_state.params, ema_value)
  new_state = new_state.replace(ema_params=new_ema)

  return new_state, metrics

#######################################################
# Sampling and Metrics
#######################################################

def sample_step(variable, sample_idx, model, rng_init, device_batch_size, config):
  """
  sample_idx: each random sampled image corrresponds to a seed
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  images = generate(variable, model, rng_sample, n_sample=device_batch_size, config=config)
  images = images.transpose(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
  return images


def run_p_sample_step(p_sample_step, state, sample_idx, latent_manager, ema=True):
  variable = {"params": state.ema_params if ema else state.params}
  latent = p_sample_step(variable, sample_idx=sample_idx)
  latent = latent.reshape(-1, *latent.shape[2:])

  # Decode
  samples = latent_manager.decode(latent)
  assert not jnp.any(jnp.isnan(samples)), f"There is nan in decoded samples! Latent range: {latent.min()}, {latent.max()}. nan in latent: {jnp.any(jnp.isnan(latent))}"

  samples = samples.transpose(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
  samples = 127.5 * samples + 128.0
  samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

  jax.random.normal(random.key(0), ()).block_until_ready() # dist sync
  return samples


def get_fid_evaluator(workdir, config, writer, p_sample_step, latent_manager):
  inception_net = fid_util.build_jax_inception()
  stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)
  run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager)
  
  def evaluator(state, epoch):
    log_for_0('Eval fid at epoch: {}'.format(epoch))

    samples_all = sample_util.generate_fid_samples(
          state, workdir, config, p_sample_step, run_p_sample_step_inner
    )
    mu, sigma = fid_util.compute_stats(samples_all, inception_net)
    fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    log_for_0(f'FID w/ EMA at {samples_all.shape[0]} samples: {fid_score}')
    
    writer.write_scalars(epoch+1, {'FID_ema': fid_score})
    writer.flush()
  return evaluator

#######################################################
# Main
#######################################################

def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:
  ########### Initialize ###########
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

  rng = random.key(config.training.seed)
  image_size = config.dataset.image_size

  log_for_0('config.training.batch_size: {}'.format(config.training.batch_size))

  if config.training.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.training.batch_size // jax.process_count()
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

  ########### Create DataLoaders ###########
  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
  )
  log_for_0('Steps per Epoch: {}'.format(steps_per_epoch))

  ########### Create Model ###########
  model_config = config.model.to_dict()
  model_str = model_config.pop('cls')

  model = MeanFlow(
    model_str=model_str,
    model_config=model_config,
    **config.sampling,
    **config.method,
  )

  ########### Create Train State ###########
  base_lr = config.training.learning_rate
  state = create_train_state(rng, config, model, image_size, lr_value=base_lr)
  if config.load_from is not None:
    state = restore_checkpoint(state, config.load_from)
  
  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch
  
  state = jax_utils.replicate(state)
  ema_fn = ema_schedules(config)

  latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size, image_size)

  p_train_step = jax.pmap(
      partial(
        train_step_with_vae, 
        rng_init=rng, 
        config=config, 
        lr=base_lr,
        ema_fn=ema_fn,
        latent_mnger=latent_manager,
      ),
      axis_name='batch',
  )
  train_metrics = []
  log_for_0('Initial compilation, this might take some minutes...')

  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())

  ########### Sampling ###########
  p_sample_step = jax.pmap(
    partial(sample_step, 
            model=model, 
            rng_init=random.PRNGKey(config.sampling.seed), 
            device_batch_size=config.fid.device_batch_size, 
            config=config,
    ),
    axis_name='batch'
  )

  if config.fid.on_training:
    fid_evaluator = get_fid_evaluator(workdir, config, writer, p_sample_step, latent_manager)

  if config.eval_only:
    fid_evaluator(state, epoch_offset)
    return state

  ########### Training Loop ###########
  for epoch in range(epoch_offset, config.training.num_epochs):

    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    log_for_0('epoch {}...'.format(epoch))
    
    ########### Sampling ###########
    if (epoch+1) % config.training.sample_per_epoch == 0 and config.training.get('sample_on_training', True):
      log_for_0(f'Samples at epoch {epoch}...')
      vis_sample = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager)
      vis_sample = make_grid_visualization(vis_sample, grid=4)
      writer.write_images(epoch+1, {'vis_sample': vis_sample})
      writer.flush()

    ########### Train ###########
    timer = Timer()
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    for n_batch, batch in enumerate(train_loader):
      step = epoch * steps_per_epoch + n_batch

      batch = input_pipeline.prepare_batch_data(batch) # the batch contains latent, both mean and var.
      state, metrics = p_train_step(state, batch)
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('Initial compilation completed. Reset timer.')
        compilation_time = timer.elapse_with_reset()
        log_for_0('p_train_step compiled in {:.2f}s'.format(compilation_time))

    ########### Metrics ###########
      train_metrics.append(metrics)
      if (step+1) % config.training.log_per_step == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = jax.tree_util.tree_map(lambda x: float(x.mean()), train_metrics)
        summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset() 
        summary["ep"] = epoch
        writer.write_scalars(step + 1, summary)

        log_for_0(
          'train epoch: %d, step: %d, loss: %.6f, steps/sec: %.2f',
          epoch, step, summary['loss'], summary['steps_per_second'],
        )
        train_metrics = []

    ########### Save Checkpoint ###########
    if (
      (epoch+1) % config.training.checkpoint_per_epoch == 0
      or (epoch+1) == config.training.num_epochs
    ):
      save_checkpoint(state, workdir)

    ########### FID ###########
    if (
        (epoch+1) % config.training.fid_per_epoch == 0
        or (epoch+1) == config.training.num_epochs
      ) and config.fid.on_training:
      fid_evaluator(state, epoch)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  return state
