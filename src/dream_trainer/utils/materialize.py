import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from typing_extensions import cast

from dream_trainer.utils import logger


def materialize_buffers(
    module: nn.Module, buffer_device: torch.device | str | None, uninitialized_modules: set[str]
):
    if callable(reset_method := getattr(module, "reset_parameters", None)):
        reset_method()  # type: ignore
    elif not all(False for _ in module.buffers(recurse=False)):
        uninitialized_modules.add(f"{type(module).__name__}.{type(module).__name__}")

    for name, buffer in module.named_buffers(recurse=False):
        module.register_buffer(
            name,
            buffer.to(buffer_device),
            persistent=name not in module._non_persistent_buffers_set,
        )

    for submodule in module.children():
        materialize_buffers(submodule, buffer_device, uninitialized_modules)

    return uninitialized_modules


def _has_cpu_offload(module: nn.Module) -> bool:
    if (
        isinstance(module, FSDPModule)
        and module._get_fsdp_state()._fsdp_param_group is not None
        and isinstance(
            cast(FSDPParamGroup, module._get_fsdp_state()._fsdp_param_group).offload_policy,
            CPUOffloadPolicy,
        )
    ):
        return True
    return any(_has_cpu_offload(child) for child in module.children())


def materialize_distributed_module(
    module: nn.Module,
    init_device: torch.device | str | None,
    buffer_device: torch.device | str | None,
) -> None:
    """
    Materialize a meta PyTorch module by allocating its parameters and buffers on the specified devices,
    and (re-)initializing parameters using the module's `reset_parameters()` method if available.

    Args:
        module (nn.Module): The PyTorch module to materialize.
        init_device (torch.device | str | None): The device on which to allocate parameters (e.g., 'cpu', 'cuda', or torch.device).
        buffer_device (torch.device | str | None): The device on which to allocate buffers (e.g., 'cpu', 'cuda', or torch.device).

    This function:
        - Moves all parameters to uninitialized memory on `init_device` using `to_empty`.
        - For each submodule, if it has parameters or buffers, attempts to call its `reset_parameters()` method to initialize them.
        - Moves all buffers to `buffer_device`.
        - Logs a warning if any submodules with parameters or buffers do not define a `reset_parameters()` method.
    """
    device = "cpu" if _has_cpu_offload(module) else init_device
    module.to_empty(device=device)
    uninitialized_modules = materialize_buffers(module, buffer_device, set())

    if uninitialized_modules:
        logger.warning(
            f"Parameter initialization incomplete for model {type(module).__name__}. The following modules have parameters or buffers with uninitialized"
            " memory because they don't define a `reset_parameters()` method for re-initialization:"
            f" {', '.join(uninitialized_modules)}"
        )
