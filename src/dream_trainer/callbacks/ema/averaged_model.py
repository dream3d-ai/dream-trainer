from typing import Any, Callable, Mapping, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from torch.optim.swa_utils import PARAM_LIST, get_swa_avg_fn, get_swa_multi_avg_fn
from torch.utils._foreach_utils import (
    _get_foreach_kernels_supported_devices,
    _group_tensors_by_device_and_dtype,
)


class EMA(Stateful):
    """
    EMA state that tracks a model's local parameter shards instead of cloning the module.

    This works with replicated modules and FSDP2/DTensor-backed modules because each rank
    stores and updates only the local tensor it owns for each parameter or buffer.
    """

    n_averaged: Tensor

    def __init__(
        self,
        model: nn.Module,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = None,
        multi_avg_fn: Callable[[PARAM_LIST, PARAM_LIST, Tensor | int], None] | None = None,
        device: str | torch.device | None = None,
        use_buffers: bool = False,
    ):
        if avg_fn is not None and multi_avg_fn is not None:
            raise ValueError("Only one of avg_fn and multi_avg_fn may be provided")

        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers
        self.device = torch.device(device) if device is not None else None

        self.parameters = {
            name: self._new_tensor(tensor) for name, tensor in model.named_parameters()
        }
        self.buffers = {
            name: self._new_tensor(tensor) for name, tensor in model.named_buffers()
        }
        self.n_averaged = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device if self.device is not None else None,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_averaged": self.n_averaged,
            "parameters": self.parameters,
            "buffers": self.buffers,
            "use_buffers": self.use_buffers,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._copy_tensor(self.n_averaged, state_dict["n_averaged"])

        loaded_parameters = state_dict["parameters"]
        self._validate_names("parameter", self.parameters, loaded_parameters)
        for name, tensor in loaded_parameters.items():
            self._copy_tensor(self.parameters[name], tensor)

        loaded_buffers = state_dict.get("buffers", {})
        if loaded_buffers:
            self._validate_names("buffer", self.buffers, loaded_buffers)
            for name, tensor in loaded_buffers.items():
                self._copy_tensor(self.buffers[name], tensor)

        self.use_buffers = state_dict.get("use_buffers", self.use_buffers)

    def initialize_from_model(self, model: nn.Module, *, n_averaged: int = 1):
        self.copy_from(model)
        self.n_averaged.fill_(n_averaged)

    @torch.no_grad()
    def copy_from(self, model: nn.Module):
        model_parameters = dict(model.named_parameters())
        self._validate_names("parameter", self.parameters, model_parameters)
        for name, tracked in self.parameters.items():
            self._copy_from_model_tensor(tracked, model_parameters[name])

        model_buffers = dict(model.named_buffers())
        self._validate_names("buffer", self.buffers, model_buffers)
        for name, tracked in self.buffers.items():
            self._copy_from_model_tensor(tracked, model_buffers[name])

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model_parameters = dict(model.named_parameters())
        self._validate_names("parameter", self.parameters, model_parameters)
        for name, tracked in self.parameters.items():
            self._copy_to_model_tensor(model_parameters[name], tracked)

        model_buffers = dict(model.named_buffers())
        self._validate_names("buffer", self.buffers, model_buffers)
        for name, tracked in self.buffers.items():
            self._copy_to_model_tensor(model_buffers[name], tracked)

    @torch.no_grad()
    def update_parameters(self, model: nn.Module):
        model_parameters = dict(model.named_parameters())
        self._validate_names("parameter", self.parameters, model_parameters)
        self._update_collection(self.parameters, model_parameters, use_ema=True)

        model_buffers = dict(model.named_buffers())
        self._validate_names("buffer", self.buffers, model_buffers)
        self._update_collection(self.buffers, model_buffers, use_ema=self.use_buffers)

        self.n_averaged.add_(1)

    def _new_tensor(self, tensor: Tensor) -> Tensor:
        local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        out = torch.zeros_like(
            local_tensor, device=self.device if self.device is not None else None
        )
        return out

    def _copy_tensor(self, dst: Tensor, src: Tensor):
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype), non_blocking=True)

    def _copy_from_model_tensor(self, dst: Tensor, src: Tensor):
        local_src = src.to_local() if isinstance(src, DTensor) else src
        dst.copy_(local_src.detach().to(device=dst.device, dtype=dst.dtype), non_blocking=True)

    def _copy_to_model_tensor(self, dst: Tensor, src: Tensor):
        local_dst = dst.to_local() if isinstance(dst, DTensor) else dst
        local_dst.copy_(
            src.detach().to(device=local_dst.device, dtype=local_dst.dtype),
            non_blocking=True,
        )

    def _validate_names(
        self,
        kind: str,
        tracked: dict[str, Tensor],
        current: Mapping[str, Tensor | nn.Parameter],
    ):
        if tracked.keys() != current.keys():
            raise ValueError(
                f"EMA {kind} set changed. "
                f"Tracked={tuple(tracked.keys())}, current={tuple(current.keys())}"
            )

    def _update_collection(
        self,
        tracked_tensors: Mapping[str, Tensor],
        model_tensors: Mapping[str, Tensor | nn.Parameter],
        *,
        use_ema: bool,
    ):
        tracked_detached: list[Tensor] = []
        model_detached: list[Tensor] = []

        for name, tracked in tracked_tensors.items():
            source = model_tensors[name]
            local_source = source.to_local() if isinstance(source, DTensor) else source
            local_source = local_source.detach().to(
                device=tracked.device,
                dtype=tracked.dtype,
                non_blocking=True,
            )

            if self.n_averaged.item() == 0:
                tracked.copy_(local_source, non_blocking=True)
                continue

            if not use_ema or not (
                torch.is_floating_point(tracked) or torch.is_complex(tracked)
            ):
                tracked.copy_(local_source, non_blocking=True)
                continue

            tracked_detached.append(tracked.detach())
            model_detached.append(local_source)

        if not tracked_detached:
            return

        if self.multi_avg_fn is not None or self.avg_fn is None:
            grouped_tensors = cast(
                dict[tuple[torch.device, torch.dtype], tuple[list[Tensor], Any]],
                _group_tensors_by_device_and_dtype([tracked_detached, model_detached]),  # type: ignore[arg-type]
            )
            for (device, _), ([grouped_tracked, grouped_model], _) in grouped_tensors.items():
                if self.multi_avg_fn is not None:
                    self.multi_avg_fn(
                        list(grouped_tracked),
                        list(grouped_model),
                        self.n_averaged.to(device=device),
                    )
                elif (
                    device is not None
                    and device.type in _get_foreach_kernels_supported_devices()
                ):
                    get_swa_multi_avg_fn()(
                        list(grouped_tracked),
                        list(grouped_model),
                        self.n_averaged.to(device=device),
                    )
                else:
                    avg_fn = get_swa_avg_fn()
                    n_averaged = self.n_averaged.to(device=device)
                    for tracked, model in zip(grouped_tracked, grouped_model, strict=True):
                        tracked.copy_(avg_fn(tracked, model, n_averaged))
            return

        for tracked, model in zip(tracked_detached, model_detached, strict=True):
            tracked.copy_(
                self.avg_fn(tracked, model, self.n_averaged.to(device=tracked.device)),
                non_blocking=True,
            )
