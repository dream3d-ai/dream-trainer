from __future__ import annotations

from typing import Iterable

from typing_extensions import override

from dream_trainer.utils import logger

from .callback import Callback

try:
    from torchinfo import summary  # type: ignore # noqa: F401
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "torchinfo is not installed. Please install it with `pip install dream-trainer[registry]` or `pip install torchinfo` to use the ModelSummary callback."
    ) from e


class ModelSummary(Callback):
    """
    Run torchinfo.summary on selected models immediately after configure_models, then exit.

    Args:
        model_names: Iterable of model attribute names to summarize. If None, summarize all.
        depth: Module depth to display in torchinfo.summary.
        row_settings: torchinfo row settings; defaults to ("var_names",).
        col_names: torchinfo column names; defaults to ("num_params", "params_percent").
    """

    def __init__(
        self,
        model_names: Iterable[str] | None = None,
        *,
        depth: int = 1,
        row_settings: Iterable[str] = ("var_names",),
        col_names: Iterable[str] = ("num_params", "params_percent"),
    ) -> None:
        super().__init__()
        self._model_names = tuple(model_names) if model_names is not None else None
        self._depth = depth
        self._row_settings = tuple(row_settings)
        self._col_names = tuple(col_names)

    @override
    def post_configure(self):
        named = self.trainer.named_models()
        target_names = self._model_names or tuple(named.keys())

        for name in target_names:
            if name not in named:
                logger.warning(
                    f"Model '{name}' not found in configured models: {tuple(named.keys())}"
                )
                continue

            model = named[name]
            print()
            print(f"===== torchinfo.summary: {name} ({model.__class__.__name__}) =====")
            try:
                summary(
                    model,
                    depth=self._depth,
                    row_settings=list(self._row_settings),
                    col_names=list(self._col_names),
                )
            except Exception as e:  # Fallback to a lightweight summary
                logger.warning(
                    f"torchinfo.summary failed for model '{name}' with error: {e}. Falling back to counts."
                )
                num_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Model: {name}")
                print(f"  Class: {model.__class__.__name__}")
                print(f"  Params: {num_params:,}")
                print(f"  Trainable Params: {trainable_params:,}")

        exit()
