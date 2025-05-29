import itertools

import torch
import torch.distributed
import torch.distributed.checkpoint.stateful
import torch.nn as nn

from dream_trainer.utils import logger


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
    module.to_empty(device=init_device)

    uninitialized_modules = set()
    for submodule in module.modules():
        if all(
            False
            for _ in itertools.chain(
                submodule.parameters(recurse=False), submodule.buffers(recurse=False)
            )
        ):
            # module has no parameters or buffers
            continue
        if callable(reset_method := getattr(submodule, "reset_parameters", None)):
            reset_method()
        else:
            uninitialized_modules.add(type(submodule).__name__)

        # move buffers to device
        for name, buffer in submodule.named_buffers(recurse=False):
            submodule.register_buffer(
                name,
                buffer.to(buffer_device),
                persistent=name not in submodule._non_persistent_buffers_set,
            )

    if uninitialized_modules:
        logger.warning(
            "Parameter initialization incomplete. The following modules have parameters or buffers with uninitialized"
            " memory because they don't define a `reset_parameters()` method for re-initialization:"
            f" {', '.join(uninitialized_modules)}"
        )
