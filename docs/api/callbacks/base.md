# Callback System

The callback system in Dream Trainer provides a powerful way to extend and customize the training loop without modifying the core trainer code. Callbacks can hook into various points of the training lifecycle to add functionality like logging, checkpointing, profiling, and more.

## Overview

Callbacks provide:
- Hooks at every stage of the training lifecycle
- Type-safe dependency injection for trainer mixins
- Automatic distributed training support
- State management for resumable training
- Context managers for training/validation steps

## Base Classes

### Callback

The base callback class that all callbacks inherit from:

::: dream_trainer.callbacks.Callback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - pre_launch
        - pre_configure
        - post_configure
        - pre_setup
        - post_setup
        - pre_fit
        - post_fit
        - pre_epoch
        - post_epoch
        - pre_train_epoch
        - post_train_epoch
        - pre_train_step
        - post_train_step
        - pre_optimizer_step
        - post_optimizer_step
        - pre_optimizer_zero_grad
        - post_optimizer_zero_grad
        - pre_validation_epoch
        - post_validation_epoch
        - pre_validation_step
        - post_validation_step
        - train_context
        - validation_context

### RankZeroCallback

A special callback that only executes on rank 0 in distributed training:

::: dream_trainer.callbacks.RankZeroCallback
    options:
      show_source: true
      show_bases: false

### CallbackCollection

Manages multiple callbacks efficiently:

::: dream_trainer.callbacks.CallbackCollection
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - initialize
        - state_dict
        - load_state_dict
        - append
        - refresh

## Creating Custom Callbacks

### Basic Callback

```python
from dream_trainer.callbacks import Callback
from dream_trainer.trainer import BaseTrainer

class SimpleLoggingCallback(Callback[BaseTrainer]):
    """Log training progress to a file."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
    
    def pre_fit(self):
        # Open log file when training starts
        self.file = open(self.log_file, "w")
        self.file.write(f"Training started for {self.trainer.experiment}\n")
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        # Log loss after each step
        loss = result.get("loss")
        if loss is not None:
            self.file.write(f"Step {batch_idx}: loss={loss:.4f}\n")
    
    def post_fit(self):
        # Close file when training ends
        self.file.write("Training completed\n")
        self.file.close()
```

### Callback with Dependencies

Callbacks can specify dependencies on trainer mixins using generics:

```python
from dream_trainer.callbacks import Callback
from dream_trainer.trainer.mixins import EvalMetricMixin, WandBLoggerMixin

class MetricLoggingCallback(Callback[EvalMetricMixin & WandBLoggerMixin]):
    """Log metrics to WandB after validation."""
    
    def post_validation_epoch(self, result: dict[str, Any]):
        # Access metrics from EvalMetricMixin
        metrics = self.trainer.named_metrics()
        
        # Compute and log metrics using WandBLoggerMixin
        for name, metric in metrics.items():
            value = metric.compute()
            self.trainer.log_scalar(f"val/{name}", value)
```

### Rank Zero Callback

For operations that should only run on the main process:

```python
from dream_trainer.callbacks import RankZeroCallback

class ModelSummaryCallback(RankZeroCallback[BaseTrainer]):
    """Print model summary only on rank 0."""
    
    def post_setup(self):
        # This only runs on rank 0
        total_params = sum(p.numel() for p in self.trainer.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.trainer.model.parameters() if p.requires_grad
        )
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
```

## Lifecycle Hooks

The callback system provides hooks at every stage of training:

### Initialization Phase

```python
class InitializationCallback(Callback[BaseTrainer]):
    def pre_launch(self):
        """Before distributed world is created."""
        print("Setting up environment variables")
    
    def pre_configure(self):
        """Before models/optimizers are configured."""
        print("Preparing configuration")
    
    def post_configure(self):
        """After configuration but before setup."""
        print("Configuration complete")
    
    def pre_setup(self):
        """Before models are materialized."""
        print("Beginning setup")
    
    def post_setup(self):
        """After all components are ready."""
        print("Setup complete")
```

### Training Phase

```python
class TrainingCallback(Callback[BaseTrainer]):
    def pre_fit(self):
        """Training is about to start."""
        self.start_time = time.time()
    
    def pre_epoch(self):
        """Before each epoch (train + val)."""
        self.epoch_start = time.time()
    
    def pre_train_epoch(self):
        """Before training epoch."""
        print(f"Starting epoch {self.trainer.current_epoch}")
    
    def pre_train_step(self, batch: dict[str, Any], batch_idx: int):
        """Before each training step."""
        # Modify batch if needed
        return batch
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        """After training step."""
        # Process results
        pass
    
    def post_train_epoch(self, result: dict[str, Any]):
        """After training epoch."""
        print(f"Epoch {self.trainer.current_epoch} complete")
    
    def post_epoch(self):
        """After full epoch (train + val)."""
        epoch_time = time.time() - self.epoch_start
        print(f"Epoch took {epoch_time:.2f}s")
    
    def post_fit(self):
        """Training complete."""
        total_time = time.time() - self.start_time
        print(f"Total training time: {total_time:.2f}s")
```

### Optimizer Hooks

```python
class GradientCallback(Callback[BaseTrainer]):
    def pre_optimizer_step(self, model: nn.Module, optimizer: Optimizer):
        """Before optimizer step."""
        # Could modify gradients here
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"Gradient norm: {total_norm:.4f}")
    
    def post_optimizer_step(self, model: nn.Module, optimizer: Optimizer):
        """After optimizer step."""
        # Log learning rate
        lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {lr}")
    
    def pre_optimizer_zero_grad(self, model: nn.Module, optimizer: Optimizer):
        """Before zeroing gradients."""
        pass
    
    def post_optimizer_zero_grad(self, model: nn.Module, optimizer: Optimizer):
        """After zeroing gradients."""
        pass
```

## Context Managers

Callbacks can provide context managers for training/validation:

```python
import contextlib
from typing import Any

class ProfilingCallback(Callback[BaseTrainer]):
    def __init__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
        )
    
    @contextlib.contextmanager
    def train_context(self):
        """Context for training steps."""
        with self.profiler:
            yield
    
    @contextlib.contextmanager
    def validation_context(self):
        """Context for validation steps."""
        # No profiling during validation
        yield
```

## State Management

Callbacks can save and restore state for resumable training:

```python
class StatefulCallback(Callback[BaseTrainer]):
    def __init__(self):
        self.step_count = 0
        self.best_metric = float("inf")
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        self.step_count += 1
    
    def post_validation_epoch(self, result: dict[str, Any]):
        metric = result.get("val_loss", float("inf"))
        self.best_metric = min(self.best_metric, metric)
    
    def state_dict(self) -> dict[str, Any]:
        """Save callback state."""
        return {
            "step_count": self.step_count,
            "best_metric": self.best_metric,
        }
    
    def load_state_dict(self, state_dict: dict[str, Any], trainer: BaseTrainer):
        """Restore callback state."""
        self.step_count = state_dict["step_count"]
        self.best_metric = state_dict["best_metric"]
```

## Using Callbacks

### Single Callback

```python
from dream_trainer.trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.callbacks import CallbackCollection

config = DreamTrainerConfig(
    # ... other config ...
    callbacks=CallbackCollection([
        SimpleLoggingCallback("training.log")
    ])
)

trainer = MyTrainer(config)
trainer.configure()
trainer.setup()
trainer.fit()
```

### Multiple Callbacks

```python
callbacks = CallbackCollection([
    ProgressBar(),
    ModelSummaryCallback(),
    CheckpointCallback(checkpoint_config),
    ProfileCallback(profiler),
    CustomMetricCallback(),
])

config = DreamTrainerConfig(
    # ... other config ...
    callbacks=callbacks
)
```

### Adding Callbacks Dynamically

```python
trainer = MyTrainer(config)

# Add callback after initialization
trainer.callbacks.append(NewCallback())

# Or use dict-style access
trainer.callbacks["CustomCallback"] = CustomCallback()

# Remove callback
del trainer.callbacks["CustomCallback"]
```

## Built-in Callbacks

Dream Trainer includes several built-in callbacks:

- **[ProgressBar](monitoring.md#progressbar)** - Display training progress
- **[CheckpointCallback](checkpoint.md)** - Save and restore checkpoints
- **[ProfileCallback](performance.md#profilecallback)** - Profile training performance
- **[LoggerCallback](monitoring.md#loggercallback)** - Log metrics
- **[TrainerSummary](monitoring.md#trainersummary)** - Display trainer configuration
- **[FindGraphBreaksCallback](performance.md#findgraphbreakscallback)** - Debug torch.compile
- **[OptimizeFSDP](performance.md#optimizefsdp)** - Optimize FSDP performance

## Best Practices

### 1. Use Type Hints

Always specify the trainer type for better IDE support:

```python
class MyCallback(Callback[DreamTrainer]):
    # Now self.trainer is typed as DreamTrainer
    def post_setup(self):
        models = self.trainer.named_models()  # Type-safe access
```

### 2. Handle Distributed Training

Be aware of distributed training when designing callbacks:

```python
class DistributedAwareCallback(Callback[BaseTrainer]):
    def post_train_epoch(self, result: dict[str, Any]):
        # Aggregate metrics across all processes
        loss = result["loss"]
        avg_loss = self.trainer.world.all_reduce(loss, op="mean")
        
        if self.trainer.world.is_global_zero:
            print(f"Average loss: {avg_loss}")
```

### 3. Avoid Side Effects

Don't modify trainer state in unexpected ways:

```python
# Bad - modifies training state
class BadCallback(Callback[BaseTrainer]):
    def pre_train_step(self, batch, batch_idx):
        self.trainer.global_step += 1  # Don't do this!

# Good - only observes state
class GoodCallback(Callback[BaseTrainer]):
    def post_train_step(self, result, batch_idx):
        print(f"Current step: {self.trainer.global_step}")
```

### 4. Resource Management

Clean up resources properly:

```python
class ResourceCallback(Callback[BaseTrainer]):
    def pre_fit(self):
        self.file = open("log.txt", "w")
        self.connection = create_database_connection()
    
    def post_fit(self):
        # Always clean up, even if training fails
        try:
            self.file.close()
        except:
            pass
        
        try:
            self.connection.close()
        except:
            pass
```

## See Also

- [CheckpointCallback](checkpoint.md) - Checkpointing implementation
- [Monitoring Callbacks](monitoring.md) - Progress and logging callbacks
- [Performance Callbacks](performance.md) - Profiling and optimization
- [BaseTrainer](../trainers/base.md) - Trainer implementation 