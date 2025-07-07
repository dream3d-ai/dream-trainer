import warnings
from typing import cast

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.optim import Optimizer
from typing_extensions import override

from dream_trainer.trainer.mixins.quantize import QuantizeMixin
from dream_trainer.utils import logger
from dream_trainer.utils.common import is_sm89_or_later

from .callback import Callback

try:
    from torchao.float8 import (  # type: ignore # noqa: F401
        Float8LinearConfig,
        convert_to_float8_training,
        precompute_float8_dynamic_scale_for_fsdp,
    )
    from torchao.float8.config import (
        CastConfig,
        Float8LinearRecipeName,
        ScalingGranularity,
        e4m3_dtype,
    )
except ImportError:
    raise ImportError(
        "torchao is not installed. Please install it with `pip install dream-trainer[fp8]` to use the Fp8Quantization callback."
    )


class Fp8Quantization(Callback[QuantizeMixin]):
    def __init__(self, recipe: str | Float8LinearRecipeName | None = None):
        if not is_sm89_or_later():
            raise ValueError("Native fp8 is only supported on H100+ GPUs.")

        self.recipe = Float8LinearRecipeName(recipe) if recipe else None

    #############
    # Callbacks #
    #############
    @override
    def pre_launch(self):
        if not (compile_model := self.trainer.device_parameters.compile_model):
            logger.warning("Compile model is disabled. Fp8 quantization may be slower.")

        self.trainer.device_parameters.async_tensor_parallel = compile_model

        # Suppress warnings about the 'use_reentrant' parameter in torch.utils.checkpoint.
        warnings.filterwarnings(
            "ignore",
            message=(
                "torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. "
                "In version 2.5 we will raise an exception if use_reentrant is not passed. "
                "use_reentrant=False is recommended, but if you need to preserve the current default behavior, "
                "you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants."
            ),
        )

    @override
    def post_configure(self):
        torch.set_float32_matmul_precision("high")

        # NOTE: force_recompute_fp8_weight_in_bwd
        # If the option is enabled, fp8_weight will always be re-computed in backward.
        # It's recommended to enable this flag when using FSDP.
        # Otherwise, the entire fp8_weight, instead of the sharded weight may be saved.
        # If using outer activation checkpointing context or SAC, you may disable this option
        # and handle the recomputation of fp8 weight in your customized AC context.

        if self.recipe is None or self.recipe == Float8LinearRecipeName.TENSORWISE:
            config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=not self.trainer.device_parameters.checkpoint_activations,
                round_scales_to_power_of_2=True,
            )
        else:
            config = Float8LinearConfig(
                # Default ROWWISE parameters
                cast_config_input=CastConfig(
                    scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
                ),
                cast_config_weight=CastConfig(
                    scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
                ),
                cast_config_grad_output=CastConfig(
                    scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
                ),
                round_scales_to_power_of_2=True,
                # FSDP specific parameters
                force_recompute_fp8_weight_in_bwd=not self.trainer.device_parameters.checkpoint_activations,
            )

        quantize_filters = self.trainer.quantize_module_filters()

        for module_name, module in self.trainer.named_models().items():
            if module_name in quantize_filters:
                setattr(
                    self.trainer,
                    module_name,
                    convert_to_float8_training(
                        module,
                        config=config,
                        module_filter_fn=quantize_filters[module_name],
                    ),
                )
                quantize_filters[module_name].validate()
                self.trainer._quantized_models.append(module_name)
                quantize_filters.pop(module_name)
                logger.info(f"Quantized model {module_name}")
        assert len(quantize_filters) == 0, (
            f"Not all keys in quantize_filters were used. Please make sure they are the same as variable names. Missing: {quantize_filters.keys()}"
        )

    @override
    def post_optimizer_step(self, model: nn.Module, optimizer: Optimizer):
        """
        Calculate scale dynamically for all float8 parameters.
        This should be run after the optimizer step. It performs a single all-reduce to compute the
        scales for all float8 weights.

        This callback hook assume there is one optimizer per model.
        """
        if isinstance(model, FSDPModule):
            # TODO: This could be done direclty on the optimizer param groups
            precompute_float8_dynamic_scale_for_fsdp(cast(nn.Module, model))
