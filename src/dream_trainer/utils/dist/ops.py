import hashlib
import pickle
from typing import Any, Collection, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .utils import TensorMetadata, apply_to_collection

CollectionType = TypeVar("CollectionType", bound=Collection)


def broadcast_collection(
    collection: CollectionType,
    process_group: ProcessGroup | None = None,
    device: torch.device | None = None,
    non_blocking: bool = True,
) -> CollectionType:
    rank = dist.get_rank(process_group)

    # Get the schema (shape, device, dtype) of tensors in the collection
    schema = apply_to_collection(
        collection, function=lambda t: TensorMetadata.from_tensor(t), dtype=torch.Tensor
    )

    schema = [schema]
    torch.distributed.broadcast_object_list(
        schema,
        group=process_group,
        group_src=0,
        device=device,
    )
    schema = schema[0]

    if rank == 0:
        collection = apply_to_collection(
            collection,
            function=lambda t: t.to(device, non_blocking=non_blocking),
            dtype=torch.Tensor,
        )
    else:
        collection = apply_to_collection(
            schema,
            function=lambda m: m.to_tensor(device=device),
            dtype=TensorMetadata,
        )

    apply_to_collection(
        collection,
        function=lambda t: torch.distributed.broadcast(t, group=process_group, group_src=0),
        dtype=torch.Tensor,
    )

    return collection


def global_agreement(
    obj: Any,
    *,
    group: dist.ProcessGroup | None = None,
    non_blocking: bool = False,
) -> bool:
    """
    Return True iff *obj* is identical on every rank in *group* (default=WORLD).

    Strategy
    --------
    • Scalars (int, bool, float) → 1-element tensor, min/max compare
    • Everything else  → SHA-256 digest → 32-element uint8 tensor, min/max compare

    That caps the wire payload at 32 B per rank and dodges CPython-hash pitfalls.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # single-process job

    def _scalar_tensor(x):
        # Pick the dtype that preserves bit-exactness
        if isinstance(x, bool):
            return torch.tensor([int(x)], dtype=torch.uint8)
        if isinstance(x, int):
            return torch.tensor([x], dtype=torch.int64)
        if isinstance(x, float):
            return torch.tensor([x], dtype=torch.float64)
        raise TypeError

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        payload = _scalar_tensor(obj).to(device, non_blocking=non_blocking)
    except (TypeError, OverflowError):  # non-scalar or huge int
        digest = hashlib.sha256(
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        ).digest()  # 32 bytes
        payload = torch.tensor(list(digest), dtype=torch.uint8, device=device)

    # Element-wise global min / max
    t_min, t_max = payload.clone(), payload.clone()
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN, group=group, async_op=non_blocking)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=group, async_op=non_blocking)

    return torch.equal(t_min, t_max)
