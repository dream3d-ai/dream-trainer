from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import torch.distributed as dist

from .core import create_local_process_group, is_local_leader, is_rank_zero

P = ParamSpec("P")  # For capturing parameter types
R = TypeVar("R")  # For capturing return type


def local_leader_only(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not dist.is_initialized():
            return func(*args, **kwargs)

        group = create_local_process_group()

        try:
            result_list: list[R] = [None]  # type: ignore
            if is_local_leader(group):
                result_list[0] = func(*args, **kwargs)

            dist.broadcast_object_list(result_list, src=0, group=group)
            return result_list[0]
        except KeyboardInterrupt:
            # Ensure all processes exit together
            dist.destroy_process_group(group)
            raise
        except Exception as e:
            # Handle timeout or other distributed errors
            dist.destroy_process_group(group)
            raise RuntimeError(f"Error in local_leader_only: {str(e)}") from e

    return wrapper


def rank_zero_only(func: Callable[P, R]) -> Callable[P, R | None]:
    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        if is_rank_zero():
            return func(*args, **kwargs)
        return None

    return wrapper
