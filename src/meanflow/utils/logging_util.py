import logging as _logging
import time

import jax
from absl import logging


def log_for_0(*args):
    if jax.process_index() == 0:
        logging.info(*args)


class ExcludeInfo(_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > _logging.INFO
        return True


exclude_files = [
    'orbax/checkpoint/async_checkpointer.py',
    'orbax/checkpoint/multihost/utils.py',
    'orbax/checkpoint/future.py',
    'orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py',
    'orbax/checkpoint/type_handlers.py',
    'orbax/checkpoint/metadata/checkpoint.py',
    'orbax/checkpoint/metadata/sharding.py'
]
file_filter = ExcludeInfo(exclude_files)


def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        """This do both elaspse and reset"""
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f'{self.elapse_with_reset():.2f} s'

