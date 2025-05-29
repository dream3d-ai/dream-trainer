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
    """
    Check if the required environment variables are set.
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
    @wraps(func)
    def wrapper(*args, **kwargs):
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
