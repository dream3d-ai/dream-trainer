import warnings
from typing import Any, Literal, cast

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
    from torchao.float8.config import Float8LinearRecipeName
    from torchao.quantization import float8_dynamic_activation_float8_weight
except ImportError:
    raise ImportError(
        "torchao is not installed. Please install it with `pip install dream-trainer[fp8]` to use the Fp8Quantization callback."
    )


class Fp8Quantization(Callback[QuantizeMixin]):
    def __init__(
        self,
        module_to_recipe: dict[str, Float8LinearRecipeName | Literal["inference"]],
    ):
        if not is_sm89_or_later():
            raise ValueError("Native fp8 is only supported on H100+ GPUs.")

        self.module_to_recipe = module_to_recipe

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
        quantize_filters = self.trainer.quantize_module_filters()
        assert all(
            module_name in quantize_filters for module_name in self.module_to_recipe.keys()
        ), (
            f"Not all modules in {self.module_to_recipe} have a quantize_filter. Missing: {set(self.module_to_recipe.keys()) - set(quantize_filters.keys())}"
        )
        assert all(
            module_name in self.trainer.named_models().keys()
            for module_name in self.module_to_recipe.keys()
        ), (
            f"Not all modules in {self.module_to_recipe} exist in {self.trainer.named_models().keys()}. Missing: {set(self.module_to_recipe.keys()) - set(self.trainer.named_models().keys())}"
        )

        for module_name, recipe in self.module_to_recipe.items():
            if recipe == "inference":
                # These need to be applied after the model is materialized
                continue

            if recipe == Float8LinearRecipeName.TENSORWISE:
                config = Float8LinearConfig(enable_fsdp_float8_all_gather=True)
            else:
                torch._inductor.config.emulate_precision_casts = True
                config = Float8LinearConfig.from_recipe_name(recipe)

            module = self.trainer.named_models()[module_name]

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
            logger.info(f"Quantized model {module_name}")

    @override
    def post_setup(self):
        from torchao.quantization.transform_module import _QUANTIZE_CONFIG_HANDLER

        quantize_filters = self.trainer.quantize_module_filters()
        config = float8_dynamic_activation_float8_weight()
        handler = _QUANTIZE_CONFIG_HANDLER[type(config)]

        for module_name, recipe in self.module_to_recipe.items():
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
                logger.info(f"Quantized model {module_name}")

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


def _is_compiled_module(m: nn.Module) -> bool:
    """
    Very light-weight check for a `torch.compile`-wrapper.

    At the time of writing, torch.compile returns an instance of
    `torch._dynamo.eval_frame.OptimizedModule`, which always exposes an
    `_orig_mod` attribute that points to the original (un-compiled) model.
    We key off that invariant instead of importing private classes.
    """
    return hasattr(m, "_orig_mod")


def _extract_compile_kwargs(compiled: nn.Module) -> dict:
    """
    Best-effort extraction of the kwargs that were supplied to torch.compile.

    PyTorch does not currently expose a public API for this, but the common
    OptimizedModule implementation stores:

        • _compile_dynamic         – bool
        • _compile_mode            – str   ("default", "reduce-overhead", …)
        • _compiler_fn             – backend callable or str
        • _compile_fullgraph       – bool
        • _compile_kwargs          – dict  (backend-specific flags)

    We harvest the subset that exists so that we can re-apply them when we
    re-compile.  Anything we can’t confidently retrieve is left to PyTorch’s
    defaults.
    """
    kw = {}

    # backend / mode
    backend = getattr(compiled, "_compiler_fn", None)
    if backend is not None:
        kw["backend"] = backend

    mode = getattr(compiled, "_compile_mode", None)
    if mode is not None:
        kw["mode"] = mode

    # dynamic, fullgraph and any opaque kwargs
    for attr, key in [
        ("_compile_dynamic", "dynamic"),
        ("_compile_fullgraph", "fullgraph"),
        ("_compile_kwargs", "options"),
    ]:
        val = getattr(compiled, attr, None)
        if val is not None:
            kw[key] = val

    return kw


def _quantize_for_inference(
    model: nn.Module,
    replacement_fn,
    filter_fn,
    cur_fqn: str = "",
    extra_args: tuple[Any, ...] | None = (),
) -> nn.Module:
    """
    Recursively replaces modules *in-place* for quantisation-time inference,
    while correctly handling `torch.compile` wrappers.

    • If the current module (or any of its children) is an OptimizedModule
      produced by `torch.compile`, we unwrap it, run the replacement pass on
      the **original** module, and then re-compile it with the same backend /
      mode so the caller sees an equivalently compiled graph.

    • The rest of the logic is unchanged from the original implementation.
    """
    # Unwrap `torch.compile` if present
    was_compiled = _is_compiled_module(model)
    compile_kwargs: dict = {}
    if was_compiled:
        compile_kwargs = _extract_compile_kwargs(model)
        model = model._orig_mod  # type: ignore[attr-defined]

    # Special-case for Float8Linear ➜ nn.Linear
    try:
        from torchao.float8.float8_linear import Float8Linear  # optional import
    except Exception:
        Float8Linear = ()  # type: ignore

    if isinstance(model, Float8Linear):
        # Build new nn.Linear on meta device so we don't allocate real memory.
        with torch.device("meta"):
            new_module = nn.Linear(model.in_features, model.out_features)
        new_module.weight = model.weight
        new_module.bias = model.bias
        model = new_module

    # Apply replacement at this node if it matches
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model, *extra_args)

    # Otherwise recurse into children
    else:
        for name, child in list(model.named_children()):
            new_child = _quantize_for_inference(
                child,
                replacement_fn,
                filter_fn,
                f"{cur_fqn}{name}.",
                extra_args,
            )
            if new_child is not child and new_child is not None:
                setattr(model, name, new_child)

    # Re-compile if we started with a compiled wrapper
    if was_compiled:
        model = torch.compile(model, **compile_kwargs)

    return model
