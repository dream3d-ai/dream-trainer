"""Generic modifiers shipped with dream-trainer.

Each function in this module is registered as a CLI flag by
[dream_trainer.utils.cli.cli][]. They are intentionally generic: none of
them reach outside of the trainer config, its `device_parameters`, its
`training_parameters`, or the built-in callbacks shipped with
dream-trainer itself.

Downstream projects can add more modifiers by importing
`dream_trainer.utils.modifiers.register_modifier` and applying it to
additional functions; those registrations merge into the same global
[MODIFIERS][dream_trainer.utils.modifiers.MODIFIERS] registry.
"""

from dream_trainer import DreamTrainerConfig

from .register import register_modifier


@register_modifier("")
def identity(config: DreamTrainerConfig):
    """No-op modifier.

    Registered under the empty string so it is excluded from the generated
    CLI flags (see `add_modifiers_to_signature` in the CLI module) while
    still satisfying lookups that expect every trainer class to have at
    least one modifier registered.
    """


@register_modifier()
def no_compile(config: DreamTrainerConfig, compile_: bool = False):
    """Disable `torch.compile` and anything that depends on it.

    Also disables async tensor parallelism (which requires compilation) and
    compiled autograd.
    """
    config.device_parameters.async_tensor_parallel = False
    config.device_parameters.compile_model = False
    config.device_parameters.compiled_autograd = False


@register_modifier()
def no_log(config: DreamTrainerConfig, log: bool = True):
    """Disable all logging backends (wandb, tensorboard, etc.)."""
    config.logging_parameters.enabled = False


@register_modifier()
def no_ckpt(config: DreamTrainerConfig):
    """Disable checkpoint saving and async checkpointing."""
    config.callbacks.pop("AsyncCheckpointCallback")
    config.callbacks.pop("CheckpointCallback")


@register_modifier()
def no_sanity(config: DreamTrainerConfig):
    """Skip the sanity validation pass that normally runs before `fit`."""
    config.training_parameters.num_sanity_val_steps = 0


@register_modifier()
def ckpt_acts(config: DreamTrainerConfig):
    """Enable activation checkpointing to trade compute for memory."""
    config.device_parameters.checkpoint_activations = True


@register_modifier()
def cpu_offload(config: DreamTrainerConfig):
    """Offload optimizer state (and where configured, parameters) to CPU."""
    config.device_parameters.cpu_offload = True


@register_modifier()
def single_device(config: DreamTrainerConfig):
    """Force single-device training while keeping existing knobs.

    Preserves `compile_model`, `cpu_offload`, and `checkpoint_activations`
    from the original `device_parameters` so those remain opt-in.
    """
    from dream_trainer.configs import DeviceParameters

    config.device_parameters = DeviceParameters.SINGLE_DEVICE(
        config.device_parameters.compile_model,
        config.device_parameters.cpu_offload,
        config.device_parameters.checkpoint_activations,
    )


@register_modifier()
def detect_anomaly(config: DreamTrainerConfig):
    """Enable `torch.autograd.set_detect_anomaly(True)` for NaN debugging."""
    import torch

    torch.autograd.set_detect_anomaly(True)


@register_modifier()
def warn_on_sync(config: DreamTrainerConfig):
    """Warn whenever a device-to-host synchronisation happens.

    Useful for tracking down `.item()` or `.cpu()` calls that silently
    break CUDA graph capture or introduce iteration stalls.
    """
    import torch

    torch.cuda.set_sync_debug_mode("warn")


@register_modifier()
def no_fp8(config: DreamTrainerConfig):
    """Disable FP8 quantization.

    No-ops (with a warning) if the Fp8Quantization callback is not
    installed or already absent from the config.
    """
    from loguru import logger

    if config.callbacks.pop("Fp8Quantization") is None:
        logger.warning("Fp8Quantization callback not found, skipping")


@register_modifier()
def display_fsdp_call_order(config: DreamTrainerConfig):
    """Print the forward-call order observed by the `OptimizeFSDP` callback.

    Useful for debugging FSDP collective ordering issues. Requires the
    `OptimizeFSDP` callback to already be present in the config.
    """
    from typing import cast

    from dream_trainer.callbacks import OptimizeFSDP

    assert "OptimizeFSDP" in config.callbacks, "OptimizeFSDP callback not found"

    cast(OptimizeFSDP, config.callbacks["OptimizeFSDP"]).display = True


@register_modifier()
def force_ddp(config: DreamTrainerConfig):
    """Force DDP on a single device. Useful for debugging parallel collectives."""
    config.device_parameters.force_ddp = True


@register_modifier()
def force_fsdp(config: DreamTrainerConfig):
    """Force FSDP on a single device. Useful for debugging parallel collectives."""
    config.device_parameters.force_fsdp = True
