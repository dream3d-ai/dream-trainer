from typing import Any

import torch

from dream_trainer.trainer.mixins.eval_metric import EvalMetricMixin
from dream_trainer.trainer.mixins.loggers import LoggerMixin

from ..callback import Callback

LoggerEvalMetricMixin = type("LoggerEvalMetricMixin", (LoggerMixin, EvalMetricMixin), {})


def filter_logs(result: dict[str, Any]) -> dict[str, Any]:
    """Filter out non-numeric values from the result dictionary."""
    filter_input = lambda value: isinstance(value, (int, float)) or (
        isinstance(value, torch.Tensor) and value.squeeze().ndim == 0
    )
    return {k: v for k, v in result.items() if filter_input(v)}


class MetricLoggerCallback(Callback[LoggerEvalMetricMixin]):
    _dependency = (LoggerMixin, EvalMetricMixin)

    def post_validation_epoch(self, result: dict[str, Any]):
        val_metrics = self.trainer.named_metrics() or {}
        metric_dict = {
            f"{title}/{name}": value
            for title, metrics in val_metrics.items()
            for name, value in metrics.compute().items()
        }
        for metrics in val_metrics.values():
            metrics.reset()

        self.trainer.log_dict({**metric_dict, **filter_logs(result)})
