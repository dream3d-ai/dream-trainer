"""Entrypoint utilities for distributed training.

This module provides utilities for automatically setting up distributed training
environments. It handles both single-process and multi-process scenarios,
automatically detecting and configuring the necessary environment variables
for PyTorch distributed training.

The main component is the `entrypoint` decorator which can be applied to
training functions to ensure they run in a properly configured distributed
environment.
"""

import os
import random
import sys
from functools import wraps
from typing import Any, Callable, TypeVar

import torch
from torch.distributed.elastic.multiprocessing.api import DefaultLogsSpecs
from torch.distributed.launcher.api import LaunchConfig, launch_agent

from dream_trainer.utils import logger

F = TypeVar("F", bound=Callable[..., Any])


def has_distributed_environment() -> bool:
    """Check if the required distributed training environment variables are set.

    This function verifies that all necessary environment variables for PyTorch
    distributed training are present in the current environment.

    Returns:
        bool: True if all required environment variables (RANK, LOCAL_RANK,
            WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are set, False otherwise.
    """
    required_env_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]
    return all(os.environ.get(var) is not None for var in required_env_vars)


def entrypoint(func: F) -> F:
    """Decorator that ensures a function runs in a properly configured distributed environment.

    This decorator automatically detects whether the code is running in a distributed
    environment. If not, it sets up the necessary environment for distributed training:

    - For single GPU: Sets up environment variables for single-process training
    - For multiple GPUs: Launches multiple processes using torch.distributed.launcher

    The decorator handles:
    - Automatic detection of available CUDA devices
    - Random port selection for distributed communication
    - Process launching for multi-GPU scenarios
    - Environment variable configuration

    Args:
        func: The function to be decorated. This should be your main training
            function that expects to run in a distributed environment.

    Returns:
        The wrapped function that will automatically set up and run in a
        distributed environment.

    Example:
        ```python
        @entrypoint
        def train():
            # Your training code here
            # Can safely use torch.distributed APIs
            pass

        if __name__ == "__main__":
            train()  # Will automatically set up distributed environment
        ```
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function that handles distributed environment setup.

        Args:
            *args: Positional arguments to pass to the decorated function.
            **kwargs: Keyword arguments to pass to the decorated function.

        Returns:
            The return value of the decorated function when running in an
            existing distributed environment. For newly launched distributed
            processes, this function doesn't return as the processes are
            managed by the launcher.
        """
        if has_distributed_environment():
            logger.info("Found distributed environment")

            return func(*args, **kwargs)

        from .names import generate_friendly_name

        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        world_size = len(cuda_devices.split(",")) if cuda_devices else torch.cuda.device_count()

        addr = "localhost"
        port = random.randint(10000, 65535)
        run_id = generate_friendly_name()

        if world_size == 1:
            logger.info("Running single process environment")
            os.environ.update(
                {
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "WORLD_SIZE": "1",
                    "MASTER_ADDR": addr,
                    "MASTER_PORT": str(port),
                    "TORCHELASTIC_RUN_ID": run_id,
                }
            )
            func(*args, **kwargs)

        else:
            logger.info(
                f"No distributed environment found, starting {world_size} local processes"
            )
            launch_agent(
                config=LaunchConfig(
                    min_nodes=1,
                    max_nodes=1,
                    nproc_per_node=world_size,
                    rdzv_backend="c10d",
                    rdzv_endpoint=f"{addr}:{port}",
                    run_id=run_id,
                    max_restarts=0,
                    logs_specs=DefaultLogsSpecs(local_ranks_filter={0}),
                ),
                entrypoint=sys.executable,
                args=sys.orig_argv[1:],
            )

    return wrapper  # type: ignore
