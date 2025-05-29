import contextlib
import functools
import os
import random
from contextlib import contextmanager
from typing import Callable, TypeVar

import torch
from torch._utils import _get_available_device_type, _get_device_module


@functools.lru_cache(None)
def is_sm89_or_later():
    """
    Check if the current CUDA device is SM 8.9 (H100 or later).

    Returns:
        bool: True if CUDA is available and the device capability is >= (8, 9), else False.
    """
    # Float8 is only supported on H100+ GPUs
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def numpy_is_installed() -> bool:
    """
    Check if numpy is installed.
    """
    try:
        import numpy  # noqa: F401 # type: ignore
    except ImportError:
        return False
    return True


def seed_everything(seed: int):
    """
    Seed all relevant random number generators to ensure reproducibility.

    This function sets the seed for Python's `random` module, PyTorch (CPU and CUDA),
    and NumPy (if installed).

    Args:
        seed (int): The seed value to use for all RNGs.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if numpy_is_installed():
        import numpy  # noqa: F401 # type: ignore

        numpy.random.seed(seed)


T = TypeVar("T")


@contextmanager
def default_dtype(dtype: torch.dtype):
    """
    Context manager that temporarily overrides the default dtype.
    """
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(orig_dtype)


@contextmanager
def configuration_ctx(self, names: list[str], *object_types: type[T]):
    """
    Context manager that temporarily overrides __setattr__ such that assigned objects of a
    specific type are stored in a dictionary alongside their name.
    """

    orig_setattr = self.__class__.__setattr__

    def custom_setattr(obj, name, value):
        if isinstance(value, object_types):
            names.append(name)
        orig_setattr(obj, name, value)

    self.__class__.__setattr__ = custom_setattr

    try:
        yield
    finally:
        self.__class__.__setattr__ = orig_setattr


@contextmanager
def stacked_context(contexts: list[contextlib._GeneratorContextManager]):
    """
    Context manager that enters multiple context managers in a stack,
    ensuring they are all properly exited in reverse order.

    Args:
        contexts (list[contextlib.AbstractContextManager]):
            A list of context managers to be entered.

    Yields:
        None
    """
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


def has_distributed_environment() -> bool:
    """
    Check if the required environment variables are set.
    """
    required_env_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    return all(os.environ.get(var) is not None for var in required_env_vars)


def get_experiment_name():
    if (run_id := os.getenv("TORCHELASTIC_RUN_ID")) is not None:
        return run_id

    from .names import generate_friendly_name

    return generate_friendly_name()


def background(func: Callable):
    """
    Decorator to run a function in a background thread.

    NOTE: This will throw away any return values.
    NOTE: Be careful with side effecting functions.
    """
    from functools import wraps
    from threading import Thread

    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper
