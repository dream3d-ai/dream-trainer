import warnings

from torch.distributed.fsdp import FSDPModule
from torchao.float8 import convert_to_float8_training, precompute_float8_dynamic_scale_for_fsdp
from torchao.quantization import float8_dynamic_activation_float8_weight
from torchao.quantization.transform_module import _QUANTIZE_CONFIG_HANDLER
from typing_extensions import override

from dream_trainer.trainer.mixins.quantize import QuantizeMixin
from dream_trainer.utils import logger
from dream_trainer.utils.common import is_sm89_or_later

from ..callback import Callback
from .types import Fp8QuantizeConfig
from .utils import _quantize_for_inference


class Fp8Quantization(Callback[QuantizeMixin]):
    def __init__(self, model_to_recipe: dict[str, Fp8QuantizeConfig | str]):
        self.model_to_recipe = {
            model_name: Fp8QuantizeConfig(recipe) if isinstance(recipe, str) else recipe
            for model_name, recipe in model_to_recipe.items()
        }

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
        quantize_filters = self.trainer.quantize_module_filters()
        assert all(
            module_name in quantize_filters for module_name in self.model_to_recipe.keys()
        ), (
            f"Not all modules in {self.model_to_recipe} have a quantize_filter. Missing: {set(self.model_to_recipe.keys()) - set(quantize_filters.keys())}"
        )
        assert all(
            module_name in self.trainer.named_models().keys()
            for module_name in self.model_to_recipe.keys()
        ), (
            f"Not all modules in {self.model_to_recipe} exist in {self.trainer.named_models().keys()}. Missing: {set(self.model_to_recipe.keys()) - set(self.trainer.named_models().keys())}"
        )

        for module_name, config in self.model_to_recipe.items():
            if config.recipe == "inference":
                # These need to be applied after the model is materialized
                continue
            else:
                config, default_filter = config.to_config()

            module = self.trainer.named_models()[module_name]
            quantize_filter = quantize_filters[module_name] + default_filter

            setattr(
                self.trainer,
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
            model = self.trainer.named_models()[module_name]

            if (
                module_name in self.trainer._quantized_models
                and config.recipe == "tensorwise"
                and isinstance(model, FSDPModule)
            ):
                optimizers = self.trainer.get_optimizers_by_model(module_name)
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
        quantize_filters = self.trainer.quantize_module_filters()
        config = float8_dynamic_activation_float8_weight()
        handler = _QUANTIZE_CONFIG_HANDLER[type(config)]  # type: ignore

        for module_name, recipe in self.model_to_recipe.items():
            if recipe == "inference":
                module = self.trainer.named_models()[module_name]
                _quantize_for_inference(
                    module,
                    handler,
                    filter_fn=quantize_filters[module_name],
                    extra_args=(config,),
                )
                quantize_filters[module_name].validate()
                self.trainer._quantized_models.append(module_name)
                logger.info(f"Quantized model {module_name} for inference")
