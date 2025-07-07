# Callbacks Guide

This guide explains how to use and create callbacks in Dream Trainer.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Built-in Callbacks](#built-in-callbacks)
- [Creating Callbacks](#creating-callbacks)
- [Callback Collection](#callback-collection)
- [Best Practices](#best-practices)

## Basic Usage

Callbacks are a way to extend the trainer's functionality without modifying its code. They are called at specific points during training.

### Adding Callbacks

Add callbacks to your trainer configuration:

```python
from dream_trainer import DreamTrainerConfig
from dream_trainer.callbacks import (
    LoggerCallback,
    ProgressBar,
    CallbackCollection
)

config = DreamTrainerConfig(
    # ... other settings ...
    callbacks=CallbackCollection([
        LoggerCallback(),  # Logs metrics to console/WandB
        ProgressBar(),    # Shows training progress
    ])
)
```

### Callback Order

Callbacks are executed in the order they are added. You can control the order:

```python
callbacks = CallbackCollection([
    LoggerCallback(),     # First: log metrics
    ProgressBar(),       # Second: show progress
    CheckpointCallback() # Third: save checkpoints
])
```

## Built-in Callbacks

### LoggerCallback

Logs metrics to console and/or WandB:

```python
from dream_trainer.callbacks import LoggerCallback

logger = LoggerCallback(
    log_every_n_steps=100,  # Log every 100 steps
    log_every_n_epochs=1,   # Log every epoch
    log_metrics=True,       # Log metrics
    log_gradients=False,    # Don't log gradients
    log_parameters=False    # Don't log parameters
)
```

### ProgressBar

Shows training progress:

```python
from dream_trainer.callbacks import ProgressBar

progress = ProgressBar(
    refresh_rate=10,        # Update every 10 steps
    show_epoch=True,        # Show epoch number
    show_step=True,         # Show step number
    show_metrics=True       # Show metrics
)
```

### CheckpointCallback

Saves model checkpoints:

```python
from dream_trainer.callbacks import CheckpointCallback

checkpoint = CheckpointCallback(
    monitor="val_loss",     # Metric to monitor
    mode="min",            # Minimize metric
    save_top_k=3,          # Keep best 3 checkpoints
    save_last=True,        # Always save latest
    every_n_epochs=1       # Save every epoch
)
```

### EarlyStoppingCallback

Stops training when metric stops improving:

```python
from dream_trainer.callbacks import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    monitor="val_loss",     # Metric to monitor
    mode="min",            # Minimize metric
    patience=5,            # Wait 5 epochs
    min_delta=0.001        # Minimum change
)
```

### LearningRateMonitor

Logs learning rate changes:

```python
from dream_trainer.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(
    logging_interval="step",  # Log every step
    log_momentum=True        # Log momentum too
)
```

## Creating Callbacks

### Basic Callback

Create a custom callback by extending `Callback`:

```python
from dream_trainer.callbacks import Callback

class MyCallback(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        """Called after each training batch"""
        # Access trainer state
        current_epoch = trainer.current_epoch
        current_step = trainer.current_step
        
        # Access outputs
        loss = outputs["loss"]
        
        # Do something
        if loss > 10.0:
            print(f"High loss detected: {loss}")
```

### Training Hooks

Available training hooks:

```python
class MyCallback(Callback):
    def on_train_start(self, trainer):
        """Called when training starts"""
        pass
        
    def on_train_epoch_start(self, trainer):
        """Called at the start of each training epoch"""
        pass
        
    def on_train_batch_start(self, trainer, batch, batch_idx):
        """Called before each training batch"""
        pass
        
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        """Called after each training batch"""
        pass
        
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch"""
        pass
        
    def on_train_end(self, trainer):
        """Called when training ends"""
        pass
```

### Validation Hooks

Available validation hooks:

```python
class MyCallback(Callback):
    def on_validation_start(self, trainer):
        """Called when validation starts"""
        pass
        
    def on_validation_epoch_start(self, trainer):
        """Called at the start of each validation epoch"""
        pass
        
    def on_validation_batch_start(self, trainer, batch, batch_idx):
        """Called before each validation batch"""
        pass
        
    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx):
        """Called after each validation batch"""
        pass
        
    def on_validation_epoch_end(self, trainer):
        """Called at the end of each validation epoch"""
        pass
        
    def on_validation_end(self, trainer):
        """Called when validation ends"""
        pass
```

### State Management

Callbacks can maintain their own state:

```python
class StatefulCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_metric = float('inf')
        self.patience_counter = 0
        
    def on_validation_epoch_end(self, trainer):
        # Get current metric
        current_metric = trainer.get_metric("val_loss")
        
        # Update state
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Check patience
        if self.patience_counter >= 5:
            trainer.should_stop = True
```

### Accessing Trainer

Callbacks have access to the trainer instance:

```python
class TrainerAwareCallback(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        # Access trainer attributes
        model = trainer.model
        optimizer = trainer.optimizer
        current_epoch = trainer.current_epoch
        
        # Access trainer methods
        trainer.log("custom_metric", 42)
        trainer.save_checkpoint("path/to/checkpoint.pt")
```

## Callback Collection

### Adding Callbacks

Add callbacks to a collection:

```python
from dream_trainer.callbacks import CallbackCollection

callbacks = CallbackCollection([
    LoggerCallback(),
    ProgressBar(),
    MyCustomCallback()
])
```

### Removing Callbacks

Remove callbacks from a collection:

```python
# Remove by type
callbacks.remove(LoggerCallback)

# Remove by instance
callbacks.remove(my_callback)
```

### Reordering Callbacks

Change callback order:

```python
# Move to front
callbacks.move_to_front(my_callback)

# Move to back
callbacks.move_to_back(my_callback)

# Move to specific position
callbacks.move_to_position(my_callback, 2)
```

## Best Practices

### 1. Keep Callbacks Focused

Each callback should do one thing well:

```python
# Good: Single responsibility
class LossMonitor(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        if outputs["loss"] > 10.0:
            print("High loss detected")

# Bad: Multiple responsibilities
class BadCallback(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        # Monitoring
        if outputs["loss"] > 10.0:
            print("High loss detected")
        # Logging
        trainer.log("custom_metric", 42)
        # Checkpointing
        trainer.save_checkpoint("checkpoint.pt")
```

### 2. Use Type Hints

Add type hints for better IDE support:

```python
from typing import Dict, Any
import torch

class TypedCallback(Callback):
    def on_train_batch_end(
        self,
        trainer: "DreamTrainer",
        outputs: Dict[str, torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int
    ) -> None:
        pass
```

### 3. Document Callbacks

Add docstrings to explain functionality:

```python
class DocumentedCallback(Callback):
    """Monitors training metrics and logs warnings.
    
    This callback watches for:
    - High loss values
    - NaN gradients
    - Learning rate spikes
    
    Args:
        loss_threshold: Threshold for high loss warning
        lr_threshold: Threshold for learning rate warning
    """
    
    def __init__(self, loss_threshold: float = 10.0, lr_threshold: float = 1e-2):
        super().__init__()
        self.loss_threshold = loss_threshold
        self.lr_threshold = lr_threshold
```

### 4. Handle Errors

Add proper error handling:

```python
class ErrorHandlingCallback(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        try:
            # Risky operation
            self.process_outputs(outputs)
        except Exception as e:
            # Log error but don't crash
            trainer.log("callback_error", str(e))
```

### 5. Test Callbacks

Write unit tests for your callbacks:

```python
def test_my_callback():
    # Create mock trainer
    trainer = MockTrainer()
    
    # Create callback
    callback = MyCallback()
    
    # Test hook
    callback.on_train_batch_end(
        trainer,
        outputs={"loss": torch.tensor(5.0)},
        batch=torch.randn(32, 10),
        batch_idx=0
    )
    
    # Assert expected behavior
    assert trainer.logged_metrics["custom_metric"] == 42
```

### 6. Use Callback Priority

Set callback priority for execution order:

```python
class HighPriorityCallback(Callback):
    priority = 100  # Higher number = earlier execution
    
class LowPriorityCallback(Callback):
    priority = 0    # Lower number = later execution
```

### 7. Avoid Side Effects

Minimize side effects in callbacks:

```python
class CleanCallback(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        # Good: Only logging
        trainer.log("metric", outputs["loss"])
        
        # Bad: Modifying trainer state
        trainer.model.requires_grad_(False)  # Don't do this
```

### 8. Use Callback Groups

Group related callbacks:

```python
class MonitoringGroup(Callback):
    """Group of monitoring callbacks"""
    
    def __init__(self):
        super().__init__()
        self.callbacks = [
            LossMonitor(),
            GradientMonitor(),
            LearningRateMonitor()
        ]
        
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        for callback in self.callbacks:
            callback.on_train_batch_end(trainer, outputs, batch, batch_idx)
```

## Next Steps

- Explore [Examples](examples.md) to see callbacks in action
- Read about [Distributed Training](distributed.md) for multi-GPU callback considerations
- Check the [API Reference](api-reference.md) for detailed callback documentation
