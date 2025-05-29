from dataclasses import dataclass

from .dataloader import DataLoaderSetupConfigMixin, DataLoaderSetupMixin
from .models import ModelSetupConfigMixin, ModelSetupMixin
from .optimizers import OptimizerAndSchedulerSetupConfigMixin, OptimizerAndSchedulerSetupMixin


@dataclass(kw_only=True)
class SetupConfigMixin(
    DataLoaderSetupConfigMixin,
    OptimizerAndSchedulerSetupConfigMixin,
    ModelSetupConfigMixin,
): ...


class SetupMixin(
    DataLoaderSetupMixin,
    OptimizerAndSchedulerSetupMixin,
    ModelSetupMixin,
):
    """
    Encapsulates all the required setup steps for the trainer.
    """

    config: SetupConfigMixin

    def configure(self):
        self._configure_models()
        self.post_configure_models()

    def setup(self):
        self._setup_models()
        self._setup_optimizers_and_schedulers()
        self._setup_dataloaders()
