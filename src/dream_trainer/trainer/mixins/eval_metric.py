from abc import abstractmethod
from dataclasses import dataclass

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

    def _configure_metrics(self):
        self._metric_names: list[str] = []

        with configuration_ctx(self, self._metric_names, MetricCollection | Metric):
            self.configure_metrics()

        for metric in self.named_metrics().values():
            metric.to(self.world.device)

        logger.info("Setup Metrics")

    def setup(self):
        super().setup()
        self._configure_metrics()
