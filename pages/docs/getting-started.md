# Getting Started with Dream Trainer

This guide will help you get up and running with Dream Trainer quickly. We'll cover installation, basic usage, and walk through a complete example.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Your First Training Run](#your-first-training-run)
- [Multi-GPU Training](#multi-gpu-training)
- [Logging and Monitoring](#logging-and-monitoring)
- [Next Steps](#next-steps)

## Installation

### Basic Installation

Install Dream Trainer using pip:

```bash
pip install dream-trainer
```

### Optional Dependencies

Dream Trainer has several optional dependencies for additional features:

```bash
# For WandB logging
pip install dream-trainer[wandb]

# For rich progress bars
pip install dream-trainer[rich]

# For metric tracking
pip install dream-trainer[metrics]

# For FP8 quantization support
pip install dream-trainer[torchao]

# For fault tolerance
pip install dream-trainer[torchft]

# Install all optional dependencies
pip install dream-trainer[all]
```

### Development Installation

For development or to run examples:

```bash
git clone https://github.com/dream3d/dream-trainer.git
cd dream-trainer
pip install -e ".[dev]"
```

## Basic Usage

### 1. Create Your Trainer

First, create a custom trainer by extending `DreamTrainer`:

```python
from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.configs import TrainingParameters, DeviceParameters
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

class MyTrainer(DreamTrainer):
    def __init__(self, config: DreamTrainerConfig, model: nn.Module):
        super().__init__(config)
        self.model = model

    def configure_models(self):
        """Configure your model(s) here"""
        # Models are automatically moved to the correct device
        # and wrapped with distributed training wrappers
        pass

    def configure_optimizers(self):
        """Configure optimizer(s)"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

    def configure_dataloaders(self):
        """Configure train and validation dataloaders"""
        # Example dummy data
        train_data = TensorDataset(
            torch.randn(1000, 10),
            torch.randint(0, 2, (1000,))
        )
        val_data = TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 2, (100,))
        )

        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=32,
            shuffle=False
        )

        return train_loader, val_loader

    def training_step(self, batch, batch_idx):
        """Define a single training step"""
        inputs, targets = batch

        # Forward pass
        outputs = self.model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        # Backward pass (handled automatically)
        self.backward(loss)

        # Return metrics to log
        return {
            "loss": loss,
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    def validation_step(self, batch, batch_idx):
        """Define a single validation step"""
        inputs, targets = batch

        # Forward pass
        outputs = self.model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        accuracy = (preds == targets).float().mean()

        return {
            "val_loss": loss,
            "val_accuracy": accuracy
        }
```

### 2. Configure Your Training

Create a configuration for your training run:

```python
from dream_trainer.callbacks import (
    LoggerCallback,
    ProgressBar,
    CallbackCollection
)

# Create configuration
config = DreamTrainerConfig(
    # Project settings
    project="my-ml-project",
    group="classification",
    experiment="baseline-v1",

    # Device settings
    device_parameters=DeviceParameters(
        # Distributed training settings
        data_parallel_size=1,  # Number of GPUs for data parallelism
        tensor_parallel_size=1,  # Tensor parallelism degree
        pipeline_parallel_size=1,  # Pipeline parallelism degree

        # Performance settings
        compile_model=True,  # Use torch.compile
        param_dtype=torch.bfloat16,  # Mixed precision
    ),

    # Training settings
    training_parameters=TrainingParameters(
        n_epochs=10,
        train_batch_size=32,
        gradient_clip_val=1.0,
        checkpoint_activations=False,

        # Validation settings
        val_frequency=0.5,  # Validate every half epoch
        num_sanity_val_steps=2,  # Sanity check before training
    ),

    # Callbacks
    callbacks=CallbackCollection([
        LoggerCallback(),  # Logs metrics to console/WandB
        ProgressBar(),  # Shows training progress
    ])
)
```

### 3. Train Your Model

```python
# Create model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# Create trainer
trainer = MyTrainer(config, model)

if __name__ == "__main__":
    from dream_trainer.utils.entrypoint import entrypoint

    entrypoint(trainer.fit)
```

## Multi-GPU Training

Dream Trainer makes distributed training simple. To use multiple GPUs:

### Single Node, Multiple GPUs

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 train.py

# Or using the trainer directly with updated config
config = DreamTrainerConfig(
    # ... other settings ...
    device_parameters=DeviceParameters(
        data_parallel_size=4,  # Use 4 GPUs
    )
)
```

### Multiple Nodes

```bash
# On node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py

# On node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py
```

## Logging and Monitoring

### WandB Integration

To use Weights & Biases for experiment tracking:

```python
from dream_trainer.configs import WandBParameters

config = DreamTrainerConfig(
    # ... other settings ...
    wandb_parameters=WandBParameters(
        project="my-project",
        entity="my-team",
        tags=["experiment", "classification"],
        notes="Initial baseline run"
    )
)
```

### Custom Logging

You can create custom logging callbacks:

```python
from dream_trainer.callbacks import Callback

class CustomLogger(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        # Log custom metrics
        trainer.log("custom_metric", outputs["custom_metric"])
```

## Next Steps

Now that you have the basics, here are some recommended next steps:

1. Read the [Configuration Guide](configuration.md) to learn about all available options
2. Check out the [Trainer Guide](trainer-guide.md) for advanced trainer customization
3. Explore [Callbacks](callbacks.md) to extend functionality
4. Try [Distributed Training](distributed.md) for multi-GPU setups
5. Look at [Examples](examples/basic.md) for complete working code

## Common Issues

### Installation Problems

If you encounter installation issues:

1. Make sure you have Python 3.10+ installed
2. Ensure PyTorch is installed correctly for your CUDA version
3. Try installing in a fresh virtual environment

### Training Issues

Common training problems and solutions:

1. **Out of Memory**: Reduce batch size or enable gradient accumulation
2. **Slow Training**: Enable mixed precision or model compilation
3. **Poor Performance**: Check learning rate and optimizer settings

## Getting Help

If you need help:

1. Check the [documentation](index.md)
2. Look at [examples](examples/basic.md)
3. Open an issue on GitHub
4. Join our community chat
