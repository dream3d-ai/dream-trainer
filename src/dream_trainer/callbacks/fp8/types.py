from dataclasses import dataclass
from typing import Literal

import torch
import torch._inductor.config
import torch.nn as nn
from torchao.float8 import CastConfig, Float8LinearConfig
from torchao.float8.config import Float8LinearRecipeName, Float8TypeConfig, ScalingType
from torchao.float8.float8_linear import ScalingGranularity

from dream_trainer.trainer.mixins.quantize import QuantizeModuleFilter


@dataclass()
class Fp8QuantizeConfig:
    recipe: Float8LinearRecipeName | Literal["inference"] | str
    pad_inner_dim: bool = False

    def __post_init__(self):
        if self.recipe != "inference":
            self.recipe = Float8LinearRecipeName(self.recipe)

    def to_config(self) -> tuple[Float8LinearConfig, QuantizeModuleFilter]:
        type_config = Float8TypeConfig()
        e4m3_dtype = type_config.e4m3_dtype

        if self.recipe is Float8LinearRecipeName.TENSORWISE:
            return Float8LinearConfig(
                enable_fsdp_float8_all_gather=True, pad_inner_dim=self.pad_inner_dim
            ), AutoFilterForTensorwise()
        elif self.recipe is Float8LinearRecipeName.ROWWISE:
            torch._inductor.config.emulate_precision_casts = True

            cc_i = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )
            cc_w = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )
            cc_go = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )

            return Float8LinearConfig(
                cast_config_input=cc_i,
                cast_config_weight=cc_w,
                cast_config_grad_output=cc_go,
                # enable power of 2 scaling factors by default for row-wise scaling
                round_scales_to_power_of_2=True,
                pad_inner_dim=self.pad_inner_dim,
            ), AutoFilterForRowwise()
        elif self.recipe is Float8LinearRecipeName.ROWWISE_WITH_GW_HP:
            torch._inductor.config.emulate_precision_casts = True
            # output_hp = input_fp8_axiswise_dim0 @ weight_t_axiswise_dim1
            cc_i = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
            cc_w = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)

            # grad_input_hp = grad_output_fp8_axiswise_dim0 @ weight_fp8_tensorwise
            cc_go = CastConfig(
                scaling_granularity=ScalingGranularity.AXISWISE, target_dtype=e4m3_dtype
            )
            cc_w_gi = CastConfig(scaling_granularity=ScalingGranularity.TENSORWISE)

            # grad_weight_hp = input_t_hp @ grad_output_hp
            cc_i_gw = CastConfig(scaling_type=ScalingType.DISABLED)
            cc_go_gw = CastConfig(scaling_type=ScalingType.DISABLED, target_dtype=e4m3_dtype)

            return Float8LinearConfig(
                cast_config_input=cc_i,
                cast_config_weight=cc_w,
                cast_config_grad_output=cc_go,
                cast_config_input_for_grad_weight=cc_i_gw,
                cast_config_weight_for_grad_input=cc_w_gi,
                cast_config_grad_output_for_grad_weight=cc_go_gw,
                pad_inner_dim=self.pad_inner_dim,
            ), AutoFilterForRowwise()
        else:
            raise ValueError(f"No config for recipe: {self.recipe}")


class AutoFilterForRowwise(QuantizeModuleFilter):
    def __call__(self, module: nn.Module, name: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False

        # All dims must be divisible by 16 due to float8 hardware requirements.
        N, K = module.weight.shape
        if not (K % 16 == 0 and N % 16 == 0):
            return False

        # Dims below these thresholds may result in worse performance
        # (see https://github.com/pytorch/ao/tree/main/torchao/float8#rowwise-scaling)
        # Note that these benchmarks referenced for auto filtering layers were run on
        # H100 GPUs, and may not be representative of other hardware.
        if N <= 2048 or K <= 1024 or (N <= 4096 and K <= 2048):
            return False

        return True

    def validate(self):
        return True


class AutoFilterForTensorwise(QuantizeModuleFilter):
    def __call__(self, module: nn.Module, name: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False

        # All dims must be divisible by 16 due to float8 hardware requirements.
        N, K = module.weight.shape
        if not (K % 16 == 0 and N % 16 == 0):
            return False

        # Dims below these thresholds may result in worse performance
        # (see https://github.com/pytorch/ao/tree/main/torchao/float8#tensorwise-scaling)
        # Note that these benchmarks referenced for auto filtering layers were run on
        # H100 GPUs, and may not be representative of other hardware.
        if K <= 4096 and N <= 1024:
            return False

        return True

    def validate(self):
        return True
