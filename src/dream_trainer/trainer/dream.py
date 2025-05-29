from dataclasses import dataclass, field

from dream_trainer.utils.common import get_experiment_name
from dream_trainer.utils.dataloader import DataLoaderCycler

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

    def _setup_trainer_metadata(self):
        super()._setup_trainer_metadata()

        self._train_iterator = iter(
            DataLoaderCycler(
                self.train_dataloader,
                device_mesh=self.world.get_mesh("cp+tp"),
                device=self.world.device,
            )
        )
        self._val_iterator = iter(
            DataLoaderCycler(
                self.val_dataloader,
                device_mesh=self.world.get_mesh("cp+tp"),
                device=self.world.device,
            )
        )
