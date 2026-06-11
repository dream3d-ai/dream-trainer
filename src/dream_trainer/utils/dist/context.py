from contextlib import contextmanager
from typing import Literal

import torch.distributed as dist

from .core import (
    create_local_process_group,
    get_dist_local_world_size,
    get_dist_rank,
    get_dist_world_size,
    is_local_leader,
    safe_barrier,
)


@contextmanager
def local_process_group(backend: Literal["nccl", "gloo"] | None = None):
    if not dist.is_initialized():
        yield None
        return

    try:
        # Group ranks by their local_rank to identify ranks on the same node
        world_size = get_dist_world_size()
        local_size = get_dist_local_world_size()
        rank = get_dist_rank()
        node_id = rank // local_size
        ranks_in_node = list(
            range(node_id * local_size, min((node_id + 1) * local_size, world_size))
        )

        group = dist.new_group(ranks=ranks_in_node, backend=backend)
        yield group
    finally:
        dist.destroy_process_group(group)


@contextmanager
def global_process_group(backend: Literal["nccl", "gloo"] | None = None):
    if not dist.is_initialized():
        yield None
        return

    try:
        # Group ranks by their local_rank to identify ranks on the same node
        world_size = get_dist_world_size()
        ranks_in_node = list(range(world_size))

        group = dist.new_group(ranks=ranks_in_node, backend=backend)
        yield group
    finally:
        dist.destroy_process_group(group)


@contextmanager
def rank_zero_first(group: dist.ProcessGroup | None = None):
    if not dist.is_initialized():
        yield
        return

    group = group or create_local_process_group()
    rank = get_dist_rank(group)

    if rank == 0:
        yield
        safe_barrier(group)
    else:
        safe_barrier(group)
        yield


@contextmanager
def local_leader_first(
    group: dist.ProcessGroup | None = None, backend: Literal["nccl", "gloo"] | None = None
):
    if not dist.is_initialized():
        yield
        return

    group = group or create_local_process_group(backend)

    if is_local_leader(group):
        yield
        safe_barrier(group)
    else:
        safe_barrier(group)
        yield
