from dataclasses import dataclass
from typing import Callable, Collection, TypeVar

import torch


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "TensorMetadata":
        return TensorMetadata(tensor.shape, tensor.dtype)

    def to_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.empty(self.shape, device=device, dtype=self.dtype)


CollectionType = TypeVar("CollectionType", bound=Collection)


def apply_to_collection(
    collection: CollectionType, function: Callable, dtype: type | tuple[type, ...]
) -> CollectionType:
    if isinstance(collection, dtype):
        return function(collection)

    if isinstance(collection, (list, tuple)):
        return type(collection)(
            apply_to_collection(item, function, dtype) for item in collection
        )

    if isinstance(collection, dict):
        return type(collection)(
            {k: apply_to_collection(v, function, dtype) for k, v in collection.items()}
        )

    return collection
