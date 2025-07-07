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
    """Configuration class for DreamTrainer using the mixin architecture.

    This class demonstrates how to compose trainer configurations using mixins.
    Each mixin provides specific configuration fields and functionality:

    - **BaseTrainerConfig**: Core training parameters (epochs, batch size, callbacks, etc.)
    - **EvalMetricConfigMixin**: Configuration for evaluation metrics (requires torchmetrics)
    - **SetupConfigMixin**: Combines model, optimizer, scheduler, and dataloader setup configs
    - **WandBLoggerConfigMixin**: Weights & Biases logging configuration

    ## Creating Custom Trainer Configurations

    To create your own trainer configuration, inherit from `BaseTrainerConfig` and
    mix in the configuration classes for the features you need:

    ```python
    @dataclass(kw_only=True)
    class MyTrainerConfig(
        BaseTrainerConfig,          # Always required as the base
        SetupConfigMixin,           # For model/optimizer/dataloader setup
        LoggerConfigMixin,          # For generic logging capabilities
        QuantizeConfigMixin,        # For quantization support (optional)
    ):
        # Add any custom configuration fields here
        my_custom_param: float = 0.1
    ```

    ## Available Configuration Mixins

    - `SetupConfigMixin`: Aggregates model, optimizer, scheduler, and dataloader configs
    - `EvalMetricConfigMixin`: Adds torchmetrics support for evaluation
    - `LoggerConfigMixin`: Base logging configuration
    - `WandBLoggerConfigMixin`: Weights & Biases specific logging
    - `QuantizeConfigMixin`: Model quantization configuration

    The order of inheritance doesn't matter for configs since they only add fields.
    """

    experiment: str = field(default_factory=get_experiment_name)


class DreamTrainer(BaseTrainer, EvalMetricMixin, SetupMixin, WandBLoggerMixin):
    """
    Proprietary DreamTrainer demonstrating the mixin architecture pattern.

    This trainer showcases how to compose functionality using mixins to create
    a fully-featured PyTorch training system. Each mixin adds specific capabilities
    while maintaining clean separation of concerns.

    ## Mixin Architecture Overview

    The DreamTrainer uses multiple inheritance to combine functionality from various mixins:

    - **BaseTrainer**: Provides the core training loop, gradient accumulation, validation,
      and callback system. This is always the foundation of any custom trainer.

    - **SetupMixin**: Handles model initialization, parallelism (FSDP, DDP, TP, PP),
      optimizer/scheduler setup, and dataloader configuration. Composed of:
      - `ModelSetupMixin`: Model configuration, weight init, parallelism, compilation
      - `OptimizerAndSchedulerSetupMixin`: Optimizer and LR scheduler management
      - `DataLoaderSetupMixin`: Train/validation dataloader setup

    - **EvalMetricMixin**: Integrates torchmetrics for standardized evaluation metrics
      with automatic device placement and distributed synchronization.

    - **WandBLoggerMixin**: Adds Weights & Biases logging with support for scalars,
      images, videos, plots, and model watching.

    ## Creating Your Own Trainer

    To create a custom trainer, follow these steps:

    ### 1. Define Your Configuration

    ```python
    @dataclass(kw_only=True)
    class MyTrainerConfig(
        BaseTrainerConfig,
        SetupConfigMixin,
        LoggerConfigMixin,
    ):
        my_param: float = 1.0
    ```

    ### 2. Create Your Trainer Class

    ```python
    class MyTrainer(BaseTrainer, SetupMixin, LoggerMixin):
        config: MyTrainerConfig

        def __init__(self, config: MyTrainerConfig):
            super().__init__(config)
    ```

    ### 3. Implement Required Methods

    Each mixin requires certain methods to be implemented:

    **From BaseTrainer (always required):**
    - `training_step(batch, batch_idx)`: Forward pass and loss computation
    - `validation_step(batch, batch_idx)`: Validation forward pass

    **From SetupMixin sub-components:**
    - `configure_models()`: Create model instances as attributes
    - `init_weights()`: Initialize model weights
    - `configure_dataloaders()`: Return (train_loader, val_loader) tuple
    - `configure_optimizers()`: Create optimizer instances as attributes
    - `configure_schedulers()`: (Optional) Create LR schedulers

    **From SetupMixin (optional, for advanced features):**
    - `apply_tensor_parallel(tp_mesh)`: Apply tensor parallelism
    - `apply_pipeline_parallel(pp_mesh)`: Apply pipeline parallelism
    - `apply_activation_checkpointing()`: Enable gradient checkpointing
    - `apply_compile()`: Compile models with torch.compile
    - `apply_fully_shard(config)`: Apply FSDP sharding
    - `apply_replicate(dp_mesh)`: Apply DDP replication

    **From EvalMetricMixin (if used):**
    - `configure_metrics()`: Create torchmetrics instances as attributes

    **From LoggerMixin variants (if used):**
    - Various logging methods are provided, override as needed

    ### 4. Example Implementation

    ```python
    class MyTrainer(BaseTrainer, SetupMixin, EvalMetricMixin):
        config: MyTrainerConfig

        def configure_models(self):
            # Models become attributes and are auto-tracked
            self.model = MyModel(self.config.model_config)

        def init_weights(self):
            # Initialize weights after model creation
            self.model.apply(self._init_weights_fn)

        def configure_optimizers(self):
            # Optimizers become attributes and are auto-tracked
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )

        def configure_dataloaders(self):
            train_loader = DataLoader(train_dataset, ...)
            val_loader = DataLoader(val_dataset, ...)
            return train_loader, val_loader

        def configure_metrics(self):
            # Metrics become attributes and are auto-tracked
            self.accuracy = torchmetrics.Accuracy()
            self.f1 = torchmetrics.F1Score()

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)

            # Use self.backward() for proper gradient scaling
            self.backward(loss)

            # Step optimizer at the right frequency
            if not self.is_accumulating_gradients:
                grad_norm = self.step(self.model, self.optimizer)

            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)

            # Update metrics
            self.accuracy(logits, y)
            self.f1(logits, y)

            return {"val_loss": loss}
    ```

    ## Mixin Composition Rules

    1. **Order matters**: Place BaseTrainer first, then mixins in order of dependencies
    2. **Config alignment**: Trainer mixins must match their config counterparts
    3. **Method resolution**: Later mixins override earlier ones (Python MRO)
    4. **Setup order**: The `setup()` method calls setup in a specific order:
       - Models → Optimizers/Schedulers → Dataloaders → Metrics

    ## Benefits of the Mixin Architecture

    - **Modularity**: Pick only the features you need
    - **Extensibility**: Easy to add new mixins for custom functionality
    - **Reusability**: Share common patterns across different trainers
    - **Separation of Concerns**: Each mixin handles one aspect of training
    - **Type Safety**: Config classes ensure required parameters are provided
    """

    config: DreamTrainerConfig

    def __init__(self, config: DreamTrainerConfig):
        super().__init__(config)
