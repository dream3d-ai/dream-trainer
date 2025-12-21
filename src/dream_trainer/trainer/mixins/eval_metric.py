import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from dream_trainer.utils import logger
from dream_trainer.utils.common import configuration_ctx

from .setup import SetupConfigMixin, SetupMixin

try:
    from torchmetrics import Metric, MetricCollection  # type: ignore # noqa: F401
except ImportError:
    raise ImportError(
        "torchmetrics is not installed. Please install it with `pip install dream-trainer[metrics]` to use the EvalMetricSetupMixin."
    )


@dataclass(kw_only=True)
class EvalMetricConfigMixin(SetupConfigMixin): ...


class EvalMetricMixin(SetupMixin):
    config: EvalMetricConfigMixin

    @abstractmethod
    def configure_metrics(self):
        pass

    def named_metrics(self) -> dict[str, MetricCollection]:
        return {name: getattr(self, name) for name in self._metric_names}

    def get_metric(self, name: str) -> MetricCollection:
        return getattr(self, name)

    def _wrap_metric_collection(self, metrics: MetricCollection) -> None:
        """
        Wrap update/reset/compute on a MetricCollection to handle empty metrics gracefully.

        - update(): sets _has_samples = True
        - reset(): sets _has_samples = False
        - compute(): returns {} if _has_samples is False
        """
        object.__setattr__(metrics, "_has_samples", False)

        original_update: Callable[..., Any] = metrics.update
        original_reset: Callable[..., Any] = metrics.reset
        original_compute: Callable[..., dict[str, Any]] = metrics.compute

        @functools.wraps(original_update)
        def wrapped_update(*args: Any, **kwargs: Any) -> Any:
            object.__setattr__(metrics, "_has_samples", True)
            return original_update(*args, **kwargs)

        @functools.wraps(original_reset)
        def wrapped_reset(*args: Any, **kwargs: Any) -> Any:
            object.__setattr__(metrics, "_has_samples", False)
            return original_reset(*args, **kwargs)

        @functools.wraps(original_compute)
        def wrapped_compute(*args: Any, **kwargs: Any) -> dict[str, Any]:
            if not getattr(metrics, "_has_samples", False):
                return {}
            return original_compute(*args, **kwargs)

        metrics.update = wrapped_update  # type: ignore[method-assign]
        metrics.reset = wrapped_reset  # type: ignore[method-assign]
        metrics.compute = wrapped_compute  # type: ignore[method-assign]

    def _configure_metrics(self):
        self._metric_names: list[str] = []

        with configuration_ctx(self, self._metric_names, MetricCollection, Metric):
            self.configure_metrics()

        for metric in self.named_metrics().values():
            metric.to(self.world.device)
            self._wrap_metric_collection(metric)

        logger.info("Setup Metrics")

    def setup(self):
        super().setup()
        self._configure_metrics()
