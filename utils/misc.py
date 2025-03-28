import os
import random
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import numpy as np
import torch
import yaml
from torch import Tensor
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def take_annotation_from(this: Callable[P, T]) -> Callable[[Callable], Callable[P, T]]:
    def decorator(real_function: Callable[P, T]) -> Callable[P, T]:
        real_function.__doc__ = this.__doc__
        return real_function

    return decorator


def send_to_device(object: Union[Tensor, Dict, List, Tuple], device: str = "cpu") -> Union[Tensor, Dict, List, Tuple]:
    """
    Send all tensors in an object to a device.

    Args:
        object (Union[Tensor, Dict, List, Tuple]): Object containing tensors.
        device (str, optional): Device to send the tensors to.

    Returns:
        object (Union[Tensor, Dict, List, Tuple]): Object with tensors on the specified device.
    """

    if isinstance(object, Tensor):
        return object.to(device)
    elif isinstance(object, dict):
        return {k: send_to_device(v, device) for k, v in object.items()}
    elif isinstance(object, list) or isinstance(object, tuple):
        return [send_to_device(element, device) for element in object]
    else:
        return object


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def save_config(path: str, config: Dict) -> None:
    """
    Saves a dictionary with the configuration parameters to a yaml file.
    """
    with open(path, mode="w") as file:
        yaml.dump(config, file)


def load_config(path: str) -> Dict:
    """
    Reads a yaml file and returns a dictionary with the configuration parameters.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config
