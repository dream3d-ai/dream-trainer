import math
from typing import Any, Literal

import torch

from dream_trainer.trainer.base import BaseTrainer

from .callback import Callback


class EarlyStoppingCallback(Callback[BaseTrainer]):
    """Track a validation metric and request a clean stop after stagnation."""

    def __init__(
        self,
        monitor: str,
        patience: int = -1,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"Unsupported early-stopping mode: {mode}")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else -float("inf")
        self.epochs_without_improvement = 0
        self.history: list[dict[str, float]] = []

    def post_validation_epoch(self, result: dict[str, Any]) -> None:
        if self.trainer.is_sanity_validation:
            return
        if self.monitor not in result:
            raise KeyError(
                f"Early-stopping metric {self.monitor!r} missing from validation result"
            )

        value = result[self.monitor]
        metric = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        self.history.append(
            {
                "global_step": float(self.trainer.global_step),
                "epoch": float(self.trainer.current_epoch),
                self.monitor: metric,
            }
        )
        improved = math.isfinite(metric) and (
            metric < self.best_metric if self.mode == "min" else metric > self.best_metric
        )
        if improved:
            self.best_metric = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.patience >= 0 and self.epochs_without_improvement > self.patience:
            self.trainer.stop_requested = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_metric = float(state_dict["best_metric"])
        self.epochs_without_improvement = int(state_dict["epochs_without_improvement"])
        self.history = list(state_dict["history"])
