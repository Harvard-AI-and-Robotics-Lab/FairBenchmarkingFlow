import jax
from flax.training import checkpoints

from utils.logging_util import log_for_0


def restore_checkpoint(state, workdir):
  state = checkpoints.restore_checkpoint(workdir, state)
  log_for_0('Restored from checkpoint at {}'.format(workdir))
  return state


def save_checkpoint(state, workdir):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  step = int(state.step)
  log_for_0('Saving checkpoint step %d.', step)
  checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)
  log_for_0('Checkpoint step %d saved.', step)