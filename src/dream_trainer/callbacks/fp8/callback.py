import warnings

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torchao.float8 import convert_to_float8_training, precompute_float8_dynamic_scale_for_fsdp
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.transform_module import _QUANTIZE_CONFIG_HANDLER
from typing_extensions import override

from dream_trainer.trainer.mixins.quantize import QuantizeMixin
from dream_trainer.utils import logger
from dream_trainer.utils.common import is_sm89_or_later

from ..callback import Callback
from .types import AutoFilterForTensorwise, Fp8QuantizeConfig
from .utils import _quantize_for_inference


class Fp8Quantization(Callback[QuantizeMixin]):
    def __init__(self, model_to_recipe: dict[str, Fp8QuantizeConfig | str]):
        self.model_to_recipe = {
            model_name: Fp8QuantizeConfig(recipe) if isinstance(recipe, str) else recipe
            for model_name, recipe in model_to_recipe.items()
        }

    ###########
    # Helpers #
    ###########

    def _resolve_module(self, name: str) -> nn.Module:
        """Resolve a possibly-dotted module path (e.g. 'pipe.text_encoder')."""
        parts = name.split(".")
        module = self.trainer.named_models()[parts[0]]
        for part in parts[1:]:
            module = getattr(module, part)
        return module

    def _set_module(self, name: str, module: nn.Module):
        """Set a module at a possibly-dotted path."""
        parts = name.split(".")
        if len(parts) == 1:
            setattr(self.trainer, name, module)
        else:
            parent = self._resolve_module(".".join(parts[:-1]))
            setattr(parent, parts[-1], module)

    def _module_exists(self, name: str) -> bool:
        try:
            self._resolve_module(name)
            return True
        except (KeyError, AttributeError):
            return False

    def _get_quantize_filters(self):
        if hasattr(self.trainer, "quantize_module_filters"):
            return self.trainer.quantize_module_filters()
        return {}

    def _lookup_filter(self, quantize_filters, name):
        """Look up user-defined filter, falling back to the root model name."""
        if name in quantize_filters:
            return quantize_filters[name]
        root = name.split(".")[0]
        if root in quantize_filters:
            return quantize_filters[root]
        return None

    #############
    # Callbacks #
    #############
    @override
    def pre_launch(self):
        if not is_sm89_or_later():
            logger.warning(
                "Native fp8 is only supported on H100+ GPUs. Disabling fp8 quantization."
            )
            self.pop_self()

        if not (compile_model := self.trainer.device_parameters.compile_model):
            logger.warning("Compile model is disabled. Fp8 quantization may be slower.")

        self.trainer.device_parameters.async_tensor_parallel = compile_model

        if not hasattr(self.trainer, "_quantized_models"):
            self.trainer._quantized_models = []

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
        quantize_filters = self._get_quantize_filters()
        missing = {
            name: self.model_to_recipe.pop(name)
            for name in list(self.model_to_recipe.keys())
            if not self._module_exists(name)
        }
        if missing:
            logger.warning(f"Modules {set(missing.keys())} could not be resolved. Skipping.")

        for module_name, config in self.model_to_recipe.items():
            module = self._resolve_module(module_name)
            if config.recipe != "inference":
                # These need to be applied after the model is materialized
                continue
            else:
                config, default_filter = config.to_config(isinstance(module, FSDPModule))

            user_filter = self._lookup_filter(quantize_filters, module_name)
            quantize_filter = (user_filter + default_filter) if user_filter else default_filter

            self._set_module(
                module_name,
                convert_to_float8_training(
                    module,
                    config=config,
                    module_filter_fn=quantize_filter,
                ),
            )

            quantize_filter.validate()
            self.trainer._quantized_models.append(module_name)
            logger.info(f"Quantized model {module_name}")

    @override
    def post_setup(self):
        """
        Setup optimizer hooks & quantize for inference.
        """
        # Setup optimizer hooks for quantized training models
        for module_name, config in self.model_to_recipe.items():
            model = self._resolve_module(module_name)

            if module_name in self.trainer._quantized_models and config.recipe == "tensorwise":
                root_name = module_name.split(".")[0]
                try:
                    optimizers = self.trainer.get_optimizers_by_model(root_name)
                except ValueError:
                    optimizers = []

                if len(optimizers) == 0:
                    logger.warning(
                        f"{module_name} is quantized for training but has no optimizers. Skipping optimizer hook setup."
                    )
                    continue

                for optimizer in optimizers:
                    optimizer.register_step_post_hook(
                        lambda *args, **kwargs: precompute_float8_dynamic_scale_for_fsdp(model)
                    )

        # Quantize inference models
        quantize_filters = self._get_quantize_filters()
        config = Float8DynamicActivationFloat8WeightConfig()
        handler = _QUANTIZE_CONFIG_HANDLER[type(config)]  # type: ignore

        for module_name, recipe in self.model_to_recipe.items():
            if recipe.recipe == "inference":
                module = self._resolve_module(module_name)
                user_filter = self._lookup_filter(quantize_filters, module_name)
                filter_fn = (
                    user_filter if user_filter is not None else AutoFilterForTensorwise()
                )
                _quantize_for_inference(
                    module,
                    handler,
                    filter_fn=filter_fn,
                    extra_args=(config,),
                )
                filter_fn.validate()
                self.trainer._quantized_models.append(module_name)
                logger.info(f"Quantized model {module_name} for inference")
