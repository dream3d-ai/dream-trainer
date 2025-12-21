from typing import Any, Callable, Optional, Union

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
from torch.utils.checkpoint import cast


class EMA(Stateful):
    """
    DTensor-compatible averaged model for EMA.

    This class is a wrapper around the model that computes the running average of the parameters.
    It is used to compute the running average of the parameters of the model.
    """

    n_averaged: Tensor

    def __init__(
        self,
        model: nn.Module,
        avg_fn: Optional[Callable[[Tensor, Tensor, Union[Tensor, int]], Tensor]] = None,
        multi_avg_fn: Optional[
            Callable[[PARAM_LIST, PARAM_LIST, Union[Tensor, int]], None]
        ] = None,
        device: str | torch.device = "cpu",
    ):
        assert avg_fn is None or multi_avg_fn is None, (
            "Only one of avg_fn and multi_avg_fn should be provided"
        )

        self.parameters = {n: self._new_buffer(p, device) for n, p in model.named_parameters()}
        self.n_averaged = torch.tensor(0, dtype=torch.long, device=device)

        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.device = device

    def state_dict(self):
        return {
            "n_averaged": self.n_averaged,
            "parameters": self.parameters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.n_averaged = state_dict["n_averaged"]
        self.parameters = state_dict["parameters"]

    def _new_buffer(self, p: Tensor, device: str | torch.device):
        local_p = p.to_local() if isinstance(p, DTensor) else p
        out = torch.zeros_like(local_p)

        # wrap subclass in DTensor as needed
        # NOTE: local tensor may have different shapes across ranks.
        # this happens when the 1st dim is not divisible by WORLD_SIZE.
        # thus, we must supply shape (and stride) to DTensor.from_local()
        if isinstance(p, DTensor):
            out = DTensor.from_local(
                local_tensor=out,
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )

        out = out.to(device)
        return out

    def update_parameters(self, model: nn.Module):
        """Update model parameters."""

        self_param_detached: list[Tensor] = []
        model_param_detached: list[Tensor] = []
        for p_averaged, p_model in zip(
            self.parameters.values(), model.parameters(), strict=True
        ):
            p_model_ = p_model.detach().to(p_averaged.device, non_blocking=True)
            model_param_detached.append(p_model_)

            self_param_detached.append(p_averaged.detach())
            if self.n_averaged == 0:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                grouped_tensors = _group_tensors_by_device_and_dtype(
                    [self_param_detached, model_param_detached]  # type: ignore[arg-type]
                )
                for (device, _), ([self_params, model_params], _) in grouped_tensors.items():
                    self_params = cast(list[Tensor], self_params)
                    model_params = cast(list[Tensor], model_params)

                    if self.multi_avg_fn:
                        self.multi_avg_fn(self_params, model_params, self.n_averaged.to(device))

                    elif (
                        device is not None
                        and device.type in _get_foreach_kernels_supported_devices()
                    ):
                        multi_avg_fn = get_swa_multi_avg_fn()
                        multi_avg_fn(self_params, model_params, self.n_averaged.to(device))

                    else:
                        avg_fn = get_swa_avg_fn()
                        n_averaged = self.n_averaged.to(device)
                        for p_averaged, p_model in zip(self_params, model_params, strict=True):
                            p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))

            else:
                for p_averaged, p_model in zip(
                    self_param_detached, model_param_detached, strict=True
                ):
                    n_averaged = self.n_averaged.to(p_averaged.device)
                    p_averaged.detach().copy_(
                        self.avg_fn(p_averaged.detach(), p_model, n_averaged), non_blocking=True
                    )

        self.n_averaged += 1
