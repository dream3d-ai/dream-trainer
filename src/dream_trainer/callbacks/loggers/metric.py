from typing import Any

import torch

from dream_trainer.trainer.mixins.eval_metric import EvalMetricMixin
from dream_trainer.trainer.mixins.loggers import LoggerMixin
from dream_trainer.utils import logger

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

    def __init__(self):
        logger.warning(
            "MetricLoggerCallback is deprecated. Metrics are now logged automatically by the LoggerCallback."
        )

    def pre_validation_epoch(self):
        for metric in self.trainer.named_metrics().values():
            metric.reset()

    def post_validation_epoch(self, result: dict[str, Any]):
        val_metrics = self.trainer.named_metrics() or {}
        metric_dict = {
            f"{title}/{name}": value
            for title, metrics in val_metrics.items()
            for name, value in metrics.compute().items()
        }

        self.trainer.log_dict({**metric_dict, **filter_logs(result)})
