from dataclasses import dataclass

from .dataloader import DataLoaderSetupConfigMixin, DataLoaderSetupMixin
from .models import ModelSetupConfigMixin, ModelSetupMixin
from .optimizers import OptimizerAndSchedulerSetupConfigMixin, OptimizerAndSchedulerSetupMixin


@dataclass(kw_only=True)
class SetupConfigMixin(
    DataLoaderSetupConfigMixin,
    OptimizerAndSchedulerSetupConfigMixin,
    ModelSetupConfigMixin,
):
    """Configuration mixin that combines all setup-related configurations.

    This dataclass serves as a unified configuration container for all aspects of
    trainer setup, including:
    - DataLoader configuration (batch sizes, workers, etc.)
    - Optimizer and scheduler configuration
    - Model setup configuration

    By inheriting from all setup configuration mixins, this class provides a single
    point of configuration for the entire training setup process.

    Example:
        @dataclass
        class MyTrainerConfig(SetupConfigMixin):
            learning_rate: float = 1e-4
            batch_size: int = 32
            model_dim: int = 768
    """

    ...


class SetupMixin(
    DataLoaderSetupMixin,
    OptimizerAndSchedulerSetupMixin,
    ModelSetupMixin,
):
    """Orchestrates the complete setup process for the trainer.

    This mixin provides a unified interface for setting up all components required
    for training, including models, optimizers, schedulers, and dataloaders. It
    ensures these components are initialized in the correct order and with proper
    dependencies.

    The setup process is divided into two main phases:
    1. Configuration phase (`configure()`): Initializes model architecture
    2. Setup phase (`setup()`): Applies parallelism, initializes weights, and
       prepares optimizers/dataloaders

    Attributes:
        config (SetupConfigMixin): Configuration object containing all setup parameters

    Example:
        class MyTrainer(SetupMixin, DreamTrainer):
            def configure_models(self):
                self.model = MyModel(self.config)

            def configure_optimizers(self):
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config.learning_rate
                )
    """

    config: SetupConfigMixin

    def configure(self):
        """Execute the configuration phase of setup.

        This method handles the initial configuration of models before any
        parallelism or optimization is applied. It performs the following steps:

        1. Configures models on meta device for memory efficiency
        2. Calls post-configuration hooks for additional setup

        This method should be called before `setup()` to ensure models are
        properly configured before being parallelized and materialized.

        Note:
            Models are created on the meta device during this phase, meaning
            they don't consume actual memory until materialized in the setup phase.
        """
        self._configure_models()
        self.post_configure_models()

    def setup(self):
        """Execute the main setup phase for all training components.

        This method orchestrates the complete setup process in the correct order:

        1. Model setup: Applies parallelism strategies (TP, PP, FSDP), compiles
           models, and initializes weights
        2. Optimizer and scheduler setup: Creates optimizers and learning rate
           schedulers based on the configured models
        3. DataLoader setup: Initializes training and validation dataloaders

        The order is important as optimizers depend on model parameters, and
        dataloaders may need information about model parallelism for proper
        sharding.

        Note:
            This method should be called after `configure()` and before training
            begins. It handles all device placement and distributed setup.
        """
        self._setup_models()
        self._setup_optimizers_and_schedulers()
        self._setup_dataloaders()
