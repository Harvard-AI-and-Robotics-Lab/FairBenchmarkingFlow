"""Default Hyperparameter configuration."""


import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # ------------------------------------------------------------
  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  
  dataset.name = 'imgnet_latent'
  dataset.root = 'DATA_ROOT'

  dataset.num_workers = 4
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = False

  dataset.image_size = 32 
  dataset.image_channels = 4 
  dataset.num_classes = 1000
  dataset.vae = 'mse'

  # ------------------------------------------------------------
  # Training
  config.training = training = ml_collections.ConfigDict()

  training.learning_rate = 0.0001
  training.batch_size = 256

  training.num_epochs = 1000
  training.log_per_step = 100
  training.sample_per_epoch = 10
  training.checkpoint_per_epoch = 10
  training.fid_per_epoch = 10
  training.half_precision = False
  
  training.seed = 42

  training.adam_b2 = 0.95
  training.ema_val = 0.9999

  # ------------------------------------------------------------
  # MeanFlow
  config.method = method = ml_collections.ConfigDict()
  
  # Noise Distribution
  method.noise_dist = 'logit_normal'
  method.P_mean = -0.4
  method.P_std = 1.0

  # Loss
  method.data_proportion = 0.75
  method.class_dropout_prob = 0.1

  # Guidance
  method.guidance_eq = 'cfg'
  method.omega = 1.0
  method.kappa = 0.5

  # Time Interval
  method.t_start = 0.0
  method.t_end = 1.0

  # Training Dynamics
  method.norm_p = 1.0
  method.norm_eps = 0.01

  # ------------------------------------------------------------
  # model
  config.model = model = ml_collections.ConfigDict()
  model.cls = 'DiT_B_4'

  # ------------------------------------------------------------
  # Sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.seed = 0
  sampling.num_steps = 1
  sampling.schedule = 'default'
  sampling.sampling_timesteps = None
  sampling.num_classes = dataset.num_classes

  # ------------------------------------------------------------
  # FID
  config.fid = fid = ml_collections.ConfigDict()
  fid.on_training = True
  fid.num_samples = 50000
  fid.device_batch_size = 128
  fid.cache_ref = 'FID_CACHE_REF'

  # others
  config.load_from = None
  config.eval_only = False

  return config

def enforce_relations(config):
  config.sampling.num_classes = config.dataset.num_classes

def metrics():
  return [
      'train_loss',
      'eval_loss',
      'train_accuracy',
      'eval_accuracy',
      'steps_per_second',
      'train_learning_rate',
  ]
