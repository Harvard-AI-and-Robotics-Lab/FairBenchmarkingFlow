import os

import jax
from flax.training import checkpoints

from utils.logging_util import log_for_0


def restore_checkpoint(state, workdir):
    """
    Restores the model state from a checkpoint located in the specified working directory.

    Supports both:
      - Flax-style parent dirs (containing numbered checkpoint subdirs)
      - Direct Orbax checkpoint dirs (e.g. downloaded pretrained weights)
    """
    # Orbax requires absolute paths.
    workdir = os.path.abspath(workdir)

    try:
        state = checkpoints.restore_checkpoint(workdir, state)
    except FileNotFoundError:
        import orbax.checkpoint as ocp
        log_for_0("Flax restore failed; trying StandardCheckpointer at {}".format(workdir))
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(
            workdir, args=ocp.args.StandardRestore(item=state)
        )
        return restored
    log_for_0("Restored from checkpoint at {}".format(workdir))
    return state


def save_checkpoint(state, workdir):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    # Save only one copy from device 0.
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)
    log_for_0("Checkpoint step %d saved.", step)
