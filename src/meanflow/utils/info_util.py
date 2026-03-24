import utils.state_util as state_util
from utils.logging_util import log_for_0


# Function to print number of parameters
def print_params(params):
    params_flatten = state_util.flatten_state_dict(params)

    total_params = 0
    max_length = max(len(k) for k in params_flatten.keys())
    max_shape = max(len(f"{p.shape}") for p in params_flatten.values())
    max_digits = max(len(f"{p.size:,}") for p in params_flatten.values())
    log_for_0('-' * (max_length + max_digits + max_shape + 8))
    for name, param in params_flatten.items():
        layer_params = param.size
        str_layer_shape = f"{param.shape}".rjust(max_shape) 
        str_layer_params = f"{layer_params:,}".rjust(max_digits)
        log_for_0(f" {name.ljust(max_length)} | {str_layer_shape} | {str_layer_params} ")
        total_params += layer_params
    log_for_0('-' * (max_length + max_digits + max_shape + 8))
    log_for_0(f"Total parameters: {total_params:,}")
