from typing import Any, override

import torch
from torch.distributed.tensor import DTensor
from torch.optim.swa_utils import get_ema_multi_avg_fn

from dream_trainer.trainer import DreamTrainer
from dream_trainer.utils import logger

from .averaged_model import EMA
from ..callback import Callback


class EMACallback(Callback[DreamTrainer]):
    """
    Exponential Moving Average (EMA) callback for DDP/FSDP trainer models.

    This callback maintains exponential moving averages of specified models during training.
    The EMA state is tracked as detached parameter/buffer tensors keyed by name so it can
    follow composable DDP and FSDP2-wrapped models without cloning the wrapped module.

    Args:
        model_names: List of model names to apply EMA to. Model names should match
            those returned by trainer.named_models().
        decay: EMA decay factor (default: 0.999). Higher values give more weight to
            historical parameters.
        update_every_n_steps: Update EMA models every N training steps (default: 1).
        start_step: Step to start EMA updates (default: 0).
        cpu_offload: Store EMA tensors on CPU instead of the model device.
        use_buffers: Track buffers in addition to parameters. Non-floating buffers are
            copied directly from the live model instead of being averaged.

    """

    def __init__(
        self,
        model_names: list[str],
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        start_step: int = 0,
        cpu_offload: bool = False,
        use_buffers: bool = False,
    ):
        self.model_names = model_names
        self.decay = decay
        self.update_every_n_steps = update_every_n_steps
        self.start_step = start_step
        self.cpu_offload = cpu_offload
        self.use_buffers = use_buffers

        # State
        self.ema_models: dict[str, EMA] = {}
        self._validation_backup: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

    @override
    def post_setup(self):
        named_models = self.trainer.named_models()
        for model_name in self.model_names:
            if model_name not in named_models:
                raise ValueError(f"Model '{model_name}' not found in trainer.named_models()")

            model = named_models[model_name]
            ema_model = EMA(
                model,
                multi_avg_fn=get_ema_multi_avg_fn(self.decay),
                device=torch.device("cpu") if self.cpu_offload else None,
                use_buffers=self.use_buffers,
            )
            ema_model.initialize_from_model(model, n_averaged=1)

            self.ema_models[model_name] = ema_model

            logger.info(f"Created EMA model for '{model_name}' with decay={self.decay}")

    @override
    def post_train_step(self, batch: dict[str, Any], batch_idx: int):
        if (
            self.trainer.global_step >= self.start_step
            and (self.trainer.global_step + 1) % self.update_every_n_steps == 0
        ):
            for model_name in self.model_names:
                model = self.trainer.get_model(model_name)
                ema_model = self.ema_models[model_name]

                # Update EMA model parameters
                ema_model.update_parameters(model)

    @override
    def pre_validation_epoch(self):
        self._validation_backup = {}

        for model_name in self.model_names:
            model = self.trainer.get_model(model_name)
            self._validation_backup[model_name] = {
                "parameters": {
                    name: (param.to_local() if isinstance(param, DTensor) else param).detach().clone()
                    for name, param in model.named_parameters()
                },
                "buffers": {
                    name: (buffer.to_local() if isinstance(buffer, DTensor) else buffer).detach().clone()
                    for name, buffer in model.named_buffers()
                },
            }
            self.ema_models[model_name].copy_to(model)

    @override
    def post_validation_epoch(self, result: dict[str, Any]):
        for model_name in self.model_names:
            model = self.trainer.get_model(model_name)
            backup = self._validation_backup.pop(model_name)

            for name, param in model.named_parameters():
                local_param = param.to_local() if isinstance(param, DTensor) else param
                local_param.copy_(backup["parameters"][name], non_blocking=True)

            for name, buffer in model.named_buffers():
                local_buffer = buffer.to_local() if isinstance(buffer, DTensor) else buffer
                local_buffer.copy_(backup["buffers"][name], non_blocking=True)

        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_names": self.model_names,
            "decay": self.decay,
            "update_every_n_steps": self.update_every_n_steps,
            "start_step": self.start_step,
            "cpu_offload": self.cpu_offload,
            "use_buffers": self.use_buffers,
            "ema_models": {
                name: ema_model.state_dict() for name, ema_model in self.ema_models.items()
            },
        }

    def _safe_pop(self, attr: str, state_dict: dict[str, Any], raise_on_change: bool = False):
        old = getattr(self, attr)
        new = state_dict.pop(attr)

        if new != old:
            if raise_on_change:
                raise ValueError(f"Got new value for EMACallback.{attr}: {new}. Expected {old}")
            else:
                logger.warning(f"Using new value for EMACallback.{attr}: {old} -> {new}")

        setattr(self, attr, new)

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._safe_pop("model_names", state_dict, raise_on_change=True)
        self._safe_pop("decay", state_dict)
        self._safe_pop("update_every_n_steps", state_dict)
        self._safe_pop("start_step", state_dict)
        self._safe_pop("cpu_offload", state_dict)
        if "use_buffers" in state_dict:
            self._safe_pop("use_buffers", state_dict)

        ema_models = state_dict.pop("ema_models")
        for model_name, ema_model in ema_models.items():
            self.ema_models[model_name].load_state_dict(ema_model)
