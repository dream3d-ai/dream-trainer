import os
from logging import Logger, _nameToLevel
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeVar

import torch
import torch.distributed as dist
import torch.utils.data

if TYPE_CHECKING:
    from loguru import Logger as LoguruLogger

P = ParamSpec("P")  # For capturing parameter types
R = TypeVar("R")  # For capturing return type


def get_dist_world_size(group: dist.ProcessGroup | None = None) -> int:
    """
    Get the number of processes in the distributed training setup.

    Returns:
        int: The number of distributed processes if distributed training is initialized,
             otherwise 1.
    """
    if dist.is_initialized():
        return dist.get_world_size(group)
    return int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))


def get_dist_rank(group: dist.ProcessGroup | None = None) -> int:
    """
    Get the rank of the current process in the distributed training setup.

    Returns:
        int: The rank of the current process if distributed training is initialized,
             otherwise 0.
    """
    if dist.is_initialized():
        return int(dist.get_rank(group))
    return int(os.environ.get("RANK", 0))


def get_dist_local_rank(group: dist.ProcessGroup | None = None) -> int:
    return get_dist_rank(group) % get_dist_local_world_size()


def get_dist_local_world_size() -> int:
    if local_world_size := os.environ.get("LOCAL_WORLD_SIZE"):
        return int(local_world_size)
    return torch.cuda.device_count()


def get_dist_node_rank(group: dist.ProcessGroup | None = None) -> int:
    return get_dist_rank(group) % get_dist_local_world_size()


def get_dist_node_world_size() -> int:
    return get_dist_world_size() // get_dist_local_world_size()


def get_mp_world_size() -> int:
    """
    Get the number of worker processes for the current DataLoader.

    Returns:
        int: The number of worker processes if running in a DataLoader worker,
             otherwise 1.
    """
    if (worker_info := torch.utils.data.get_worker_info()) is not None:
        return worker_info.num_workers
    return 1


def get_mp_rank() -> int:
    """
    Get the rank of the current DataLoader worker process.

    Returns:
        int: The rank of the current DataLoader worker if running in a worker process,
             otherwise 0.
    """
    if (worker_info := torch.utils.data.get_worker_info()) is not None:
        return worker_info.id
    return 0


def is_rank_zero(group: dist.ProcessGroup | None = None) -> bool:
    return get_dist_rank(group) == 0


def rank_zero_log(
    message: str,
    logger: "Logger | LoguruLogger",
    log_level: Literal["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO",
) -> None:
    log_level_int = _nameToLevel[log_level]
    if is_local_leader() and get_mp_rank() == 0:
        logger.log(log_level_int, message)


def is_local_leader(group: dist.ProcessGroup | None = None) -> bool:
    return get_dist_local_rank(group) == 0


def is_mp_local_leader() -> bool:
    return get_mp_rank() == 0


def safe_barrier(group: dist.ProcessGroup | None = None) -> None:
    if not dist.is_initialized():
        return

    if dist.get_backend(group) == "nccl":
        dist.barrier(device_ids=[get_dist_rank(group)])
    else:
        dist.barrier()


def create_local_process_group(backend: Literal["nccl", "gloo"] | None = None):
    """
    Get or create a process group for ranks on the same node.
    """
    if not dist.is_initialized():
        return None

    if not hasattr(create_local_process_group, "_cache"):
        # Group ranks by their local_rank to identify ranks on the same node
        world_size = get_dist_world_size()
        local_size = get_dist_local_world_size()
        rank = get_dist_rank()
        node_id = rank // local_size
        ranks_in_node = list(
            range(node_id * local_size, min((node_id + 1) * local_size, world_size))
        )

        # Create a process group for this node
        group = dist.new_group(ranks=ranks_in_node, backend=backend)
        create_local_process_group._cache = group

    return create_local_process_group._cache


def create_global_process_group(backend: Literal["nccl", "gloo"] | None = None):
    """
    Get or create a process group for ranks across all nodes.
    """
    if not dist.is_initialized():
        return None

    if not hasattr(create_global_process_group, "_cache"):
        # Group ranks by their local_rank to identify ranks on the same node
        world_size = get_dist_world_size()
        ranks_in_node = list(range(world_size))

        # Create a process group for this node
        group = dist.new_group(ranks=ranks_in_node, backend=backend)
        create_global_process_group._cache = group

    return create_global_process_group._cache
