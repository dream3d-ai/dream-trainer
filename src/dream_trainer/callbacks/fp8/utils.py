from typing import Any, cast

import torch
import torch.nn as nn


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
    extra_args: tuple[Any, ...] = (),
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
        model = cast(nn.Module, torch.compile(model, **compile_kwargs))

    return model
