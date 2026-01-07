# Tutorial: Your First Trainer

This step-by-step tutorial guides you through building your first Dream Trainer from scratch. By the end, you'll understand the core concepts and have a working training pipeline.

## What You'll Learn

- How to structure a trainer using mixins
- How to define models, optimizers, and dataloaders
- How to implement training and validation steps
- How to run training and monitor progress

## Prerequisites

- Dream Trainer installed (`pip install dream-trainer`)
- Basic PyTorch knowledge
- A GPU (optional but recommended)

## Step 1: Understand the Architecture

Dream Trainer uses a **mixin-based architecture**. Instead of inheriting from a monolithic base class, you compose your trainer from small, focused mixins:

```
BaseTrainer          <- Core training loop
    + SetupMixin     <- Model/optimizer/dataloader setup
    + EvalMetricMixin <- Metrics tracking (optional)
    + WandBLoggerMixin <- Logging (optional)
    = YourTrainer
```

Each mixin adds specific functionality and requires you to implement certain methods.

## Step 2: Create Your Configuration

Dream Trainer uses Python dataclasses for configuration. This gives you type safety, IDE support, and composability.

```python
# train.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dream_trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin
from dream_trainer.configs import TrainingParameters, DeviceParameters


@dataclass
class MNISTConfig(BaseTrainerConfig, SetupConfigMixin):
    """Configuration for our MNIST trainer.

    BaseTrainerConfig provides:
        - training_parameters: TrainingParameters
        - device_parameters: DeviceParameters
        - callbacks: CallbackCollection

    SetupConfigMixin adds configuration for model setup.

    We add our custom fields below.
    """
    # Model architecture
    hidden_size: int = 128
    num_classes: int = 10

    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 64
```

## Step 3: Define Your Model

Create a simple neural network. Dream Trainer works with any `nn.Module`:

```python
class MNISTModel(nn.Module):
    """Simple feedforward network for MNIST classification."""

    def __init__(self, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

## Step 4: Create the Trainer

Now let's create the trainer by combining `BaseTrainer` with `SetupMixin`:

```python
class MNISTTrainer(BaseTrainer, SetupMixin):
    """Trainer for MNIST digit classification.

    By inheriting from SetupMixin, we need to implement:
        - configure_models(): Create model(s)
        - init_weights(): Initialize model weights
        - configure_optimizers(): Create optimizer(s)
        - configure_dataloaders(): Return (train_loader, val_loader)

    By inheriting from BaseTrainer, we need to implement:
        - training_step(): Forward pass and loss computation
        - validation_step(): Validation forward pass
    """
    config: MNISTConfig  # Type hint for IDE support

    # ==========================================
    # SetupMixin Required Methods
    # ==========================================

    def configure_models(self):
        """Create model(s) on meta device.

        Models are created on meta device (no memory) and
        moved to real devices during setup. This allows
        for efficient distributed initialization.
        """
        self.model = MNISTModel(
            hidden_size=self.config.hidden_size,
            num_classes=self.config.num_classes,
        )

    def init_weights(self):
        """Initialize model weights after device placement.

        This is called after the model has been moved to
        the target device(s) and any parallelism has been applied.
        """
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.model.apply(_init_weights)

    def configure_optimizers(self):
        """Create optimizer(s).

        Called after init_weights(). The model is on the
        correct device at this point.
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def configure_dataloaders(self):
        """Return (train_dataloader, val_dataloader).

        Dream Trainer handles distributed sampling automatically
        when using multiple GPUs.
        """
        # Create synthetic MNIST-like data for this tutorial
        # In practice, you'd use torchvision.datasets.MNIST
        train_x = torch.randn(10000, 1, 28, 28)
        train_y = torch.randint(0, 10, (10000,))
        val_x = torch.randn(1000, 1, 28, 28)
        val_y = torch.randint(0, 10, (1000,))

        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        return train_loader, val_loader

    # ==========================================
    # BaseTrainer Required Methods
    # ==========================================

    def training_step(self, batch, batch_idx):
        """Execute one training step.

        This method is called for each batch during training.

        Args:
            batch: A batch from train_dataloader
            batch_idx: Index of this batch in the epoch

        Returns:
            Dictionary with at least 'loss' key
        """
        images, labels = batch

        # Move data to device (Dream Trainer handles this)
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Forward pass
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)

        # Backward pass - Dream Trainer handles gradient accumulation
        self.backward(loss)

        # Step optimizer only when not accumulating gradients
        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            return {
                "loss": loss,
                "grad_norm": grad_norm,
            }

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Execute one validation step.

        Called with torch.no_grad() context automatically.

        Args:
            batch: A batch from val_dataloader
            batch_idx: Index of this batch

        Returns:
            Dictionary with validation metrics
        """
        images, labels = batch

        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Forward pass (no backward needed)
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum()
        total = labels.size(0)

        return {
            "val_loss": loss,
            "val_correct": correct,
            "val_total": total,
        }
```

## Step 5: Configure and Run Training

Now let's put it all together:

```python
def main():
    # Create configuration
    config = MNISTConfig(
        # Training settings
        training_parameters=TrainingParameters(
            n_epochs=5,
            train_batch_size=64,
            gradient_clip_val=1.0,
            val_frequency=0.5,  # Validate twice per epoch
        ),

        # Device settings
        device_parameters=DeviceParameters.SINGLE_DEVICE(),

        # Custom settings
        hidden_size=256,
        learning_rate=1e-3,
    )

    # Create trainer
    trainer = MNISTTrainer(config)

    # Start training!
    trainer.fit()


if __name__ == "__main__":
    main()
```

## Step 6: Run Your Training

```bash
# Single GPU
python train.py

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

You should see output like:

```
[2024-01-15 10:30:00] INFO: Starting training
[2024-01-15 10:30:00] INFO: Epoch 1/5
[2024-01-15 10:30:05] INFO: train/loss: 2.31, train/grad_norm: 0.45
[2024-01-15 10:30:10] INFO: Validation - val_loss: 2.15, accuracy: 0.25
...
```

## Step 7: Add Callbacks

Let's enhance our trainer with callbacks for checkpointing and progress tracking:

```python
from dream_trainer.callbacks import (
    CheckpointCallback,
    ProgressBar,
    LoggerCallback,
    CallbackCollection,
)
from dream_trainer.configs import CheckpointParameters


def main():
    # Create callbacks
    callbacks = CallbackCollection([
        # Progress bar
        ProgressBar(),

        # Log metrics every 10 batches
        LoggerCallback(log_every_n_train_batches=10),

        # Save checkpoints
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_dir="./checkpoints",
                checkpoint_every_n_epochs=1,
                keep_top_k=3,
                monitor="val_loss",
                mode="min",
            )
        ),
    ])

    config = MNISTConfig(
        training_parameters=TrainingParameters(
            n_epochs=10,
            train_batch_size=64,
        ),
        device_parameters=DeviceParameters.SINGLE_DEVICE(),
        callbacks=callbacks,  # Add callbacks to config
    )

    trainer = MNISTTrainer(config)
    trainer.fit()
```

## Step 8: Add Metrics Tracking

Use `EvalMetricMixin` for automatic metrics computation:

```python
from dream_trainer.trainer.mixins import EvalMetricMixin, EvalMetricConfigMixin
import torchmetrics


@dataclass
class MNISTConfigWithMetrics(
    BaseTrainerConfig,
    SetupConfigMixin,
    EvalMetricConfigMixin  # Add metrics config
):
    hidden_size: int = 128
    num_classes: int = 10
    learning_rate: float = 1e-3


class MNISTTrainerWithMetrics(BaseTrainer, SetupMixin, EvalMetricMixin):
    config: MNISTConfigWithMetrics

    def configure_metrics(self):
        """Configure torchmetrics for evaluation.

        Metrics are automatically moved to the correct device
        and synchronized across distributed processes.
        """
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.config.num_classes
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.config.num_classes,
            average="macro"
        )

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)

        # Update metrics - they handle accumulation automatically
        preds = logits.argmax(dim=-1)
        self.accuracy.update(preds, labels)
        self.f1_score.update(preds, labels)

        return {"val_loss": loss}

    # ... other methods remain the same
```

## Complete Example

Here's the full code in one file:

```python
"""Complete MNIST training example with Dream Trainer."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dream_trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin
from dream_trainer.configs import TrainingParameters, DeviceParameters
from dream_trainer.callbacks import (
    CheckpointCallback,
    ProgressBar,
    LoggerCallback,
    CallbackCollection,
)
from dream_trainer.configs import CheckpointParameters


# Configuration
@dataclass
class MNISTConfig(BaseTrainerConfig, SetupConfigMixin):
    hidden_size: int = 128
    num_classes: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 64


# Model
class MNISTModel(nn.Module):
    def __init__(self, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# Trainer
class MNISTTrainer(BaseTrainer, SetupMixin):
    config: MNISTConfig

    def configure_models(self):
        self.model = MNISTModel(
            hidden_size=self.config.hidden_size,
            num_classes=self.config.num_classes,
        )

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.model.apply(_init)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def configure_dataloaders(self):
        # Synthetic data - replace with real MNIST
        train_x = torch.randn(10000, 1, 28, 28)
        train_y = torch.randint(0, 10, (10000,))
        val_x = torch.randn(1000, 1, 28, 28)
        val_y = torch.randint(0, 10, (1000,))

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=self.config.batch_size * 2,
        )
        return train_loader, val_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            return {"loss": loss, "grad_norm": grad_norm}
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=-1)
        correct = (preds == y).sum()

        return {"val_loss": loss, "correct": correct, "total": y.size(0)}


def main():
    callbacks = CallbackCollection([
        ProgressBar(),
        LoggerCallback(log_every_n_train_batches=50),
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_dir="./checkpoints",
                checkpoint_every_n_epochs=1,
            )
        ),
    ])

    config = MNISTConfig(
        training_parameters=TrainingParameters(
            n_epochs=5,
            train_batch_size=64,
            val_frequency=0.5,
        ),
        device_parameters=DeviceParameters.SINGLE_DEVICE(),
        callbacks=callbacks,
        hidden_size=256,
        learning_rate=1e-3,
    )

    trainer = MNISTTrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
```

## Next Steps

Now that you've built your first trainer:

1. **[Multi-GPU Training](multi-gpu.md)**: Scale to multiple GPUs with FSDP2
2. **[Custom Components](custom-components.md)**: Create your own mixins and callbacks
3. **[Configuration Guide](../configuration.md)**: Learn all configuration options
4. **[Callbacks](../callbacks.md)**: Explore the callback system

## Troubleshooting

### Common Issues

**"No module named 'dream_trainer'"**
- Ensure Dream Trainer is installed: `pip install dream-trainer`

**"CUDA out of memory"**
- Reduce batch size in `TrainingParameters`
- Enable gradient checkpointing in `DeviceParameters`

**"Training is slow"**
- Increase `num_workers` in DataLoader
- Use `pin_memory=True` in DataLoader
- Enable mixed precision in `DeviceParameters`

## Summary

In this tutorial, you learned:

1. Dream Trainer uses mixins for composable functionality
2. Configuration is done with type-safe Python dataclasses
3. `SetupMixin` requires: `configure_models`, `init_weights`, `configure_optimizers`, `configure_dataloaders`
4. `BaseTrainer` requires: `training_step`, `validation_step`
5. Use `self.backward()` and `self.step()` for automatic gradient handling
6. Callbacks add functionality like checkpointing and progress bars

Happy training!
