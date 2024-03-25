import os
import torch


def cuda_env_device(id):
    if "MODEL_PARALLELISM" not in os.environ:
        return None
    return int(os.environ[f"DEVICE_ID{id}"])


def model_parallelism_to_device(*args):
    """Put multiple objects to cuda device for model parallelism.
    The last positional argument is device ID."""
    device = args[-1]
    args = args[:-1]

    if "MODEL_PARALLELISM" not in os.environ:
        return args if len(args) > 1 else args[0]

    new_args = []
    for arg in args:
        if arg is None:
            new_args.append(arg)
        else:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.to(device))
            if isinstance(arg, list):
                new_args.append([arg_i.to(device) for arg_i in arg])
    return new_args if len(new_args) > 1 else new_args[0]
