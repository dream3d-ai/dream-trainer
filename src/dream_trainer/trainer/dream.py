from dataclasses import dataclass, field

from dream_trainer.utils.common import get_experiment_name

from .base import BaseTrainer, BaseTrainerConfig
from .mixins import (
    EvalMetricConfigMixin,
    EvalMetricMixin,
    SetupConfigMixin,
    SetupMixin,
    WandBLoggerConfigMixin,
    WandBLoggerMixin,
)


@dataclass(kw_only=True)
class DreamTrainerConfig(
    BaseTrainerConfig, EvalMetricConfigMixin, SetupConfigMixin, WandBLoggerConfigMixin
):
    experiment: str = field(default_factory=get_experiment_name)


class DreamTrainer(BaseTrainer, EvalMetricMixin, SetupMixin, WandBLoggerMixin):
    """
    Proprietary DreamTrainer.
    """

    config: DreamTrainerConfig

    def __init__(self, config: DreamTrainerConfig):
        super().__init__(config)
