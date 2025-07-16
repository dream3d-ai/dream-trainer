# Monitoring Callbacks

Dream Trainer provides several callbacks for monitoring training progress, logging metrics, and displaying training summaries.

## ProgressBar

Displays a real-time progress bar for training and validation:

::: dream_trainer.callbacks.ProgressBar
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__

### Usage

```python
from dream_trainer.callbacks import ProgressBar

# Basic progress bar
progress = ProgressBar()

# With custom smoothing and metric display
progress = ProgressBar(
    smoothing=0.1,  # Smoothing factor for speed estimates
    metric="loss"   # Display loss in progress bar
)

# Add to trainer
callbacks = CallbackCollection([progress])
```

### Output Example

```
Epoch:     5/100 [05:23<1:32:45, 58.58s/it]
Training:  450/1000 [45%] [02:15<02:45, 3.00it/s, loss=0.453]
Validation: 50/100 [50%] [00:30<00:30, 1.67it/s]
```

## LoggerCallback

Base callback for logging training metrics:

::: dream_trainer.callbacks.LoggerCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - pre_configure
        - pre_fit
        - post_train_step
        - post_train_epoch
        - post_validation_step
        - post_validation_epoch

### Usage

```python
from dream_trainer.callbacks import LoggerCallback

# Log every batch
logger = LoggerCallback()

# Log every N batches
logger = LoggerCallback(
    log_every_n_train_batches=10,  # Log training every 10 batches
    log_every_n_val_batches=5,      # Log validation every 5 batches
    code_dir="./"                   # Directory to save code
)
```

## MediaLoggerCallback

Logs images, videos, and other media during training:

::: dream_trainer.callbacks.MediaLoggerCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false

### Usage

```python
from dream_trainer.callbacks import MediaLoggerCallback

class CustomMediaLogger(MediaLoggerCallback):
    def __init__(self, log_every_n_epochs: int = 5):
        self.log_every_n_epochs = log_every_n_epochs
    
    def post_validation_epoch(self, result: dict[str, Any]):
        if self.trainer.current_epoch % self.log_every_n_epochs == 0:
            # Generate and log sample images
            with torch.no_grad():
                samples = self.trainer.model.generate(num_samples=8)
                self.trainer.log_images(
                    samples,
                    desc="generated_samples"
                )
            
            # Log attention maps if available
            if hasattr(self.trainer.model, "get_attention_maps"):
                attention = self.trainer.model.get_attention_maps()
                self.trainer.log_images(
                    attention,
                    desc="attention_maps"
                )
```

## MetricLoggerCallback

Logs torchmetrics values (requires EvalMetricMixin):

::: dream_trainer.callbacks.MetricLoggerCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false

### Usage

```python
from dream_trainer.callbacks import MetricLoggerCallback

# Automatically log all metrics
metric_logger = MetricLoggerCallback()

# Custom metric logging
class CustomMetricLogger(MetricLoggerCallback):
    def post_validation_epoch(self, result: dict[str, Any]):
        # Log individual metrics
        for name, metric in self.trainer.named_metrics().items():
            value = metric.compute()
            self.trainer.log_scalar(f"val/metrics/{name}", value)
        
        # Log metric visualizations
        if hasattr(self.trainer, "confusion_matrix"):
            cm = self.trainer.confusion_matrix.compute()
            fig = self.plot_confusion_matrix(cm)
            self.trainer.log_plot(fig, desc="val/confusion_matrix")
```

## TrainerSummary

Displays a comprehensive summary of the trainer configuration:

::: dream_trainer.callbacks.TrainerSummary
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false

### Usage

```python
from dream_trainer.callbacks import TrainerSummary

# Add to callbacks
summary = TrainerSummary()
callbacks = CallbackCollection([summary])
```

### Output Example

```
╭─────────────────────────── Trainer Summary ────────────────────────────╮
│                                                                         │
│ Experiment: llama3-7b-finetune                                          │
│ Group: language-modeling                                                │
│ Project: research                                                       │
│                                                                         │
│ ╭───────────────────────── Model Summary ─────────────────────────╮    │
│ │ Model: LlamaModel                                                │    │
│ │ Parameters: 6,738,415,616                                        │    │
│ │ Trainable: 6,738,415,616                                         │    │
│ │ Non-trainable: 0                                                 │    │
│ ╰──────────────────────────────────────────────────────────────────╯    │
│                                                                         │
│ ╭──────────────────────── Training Config ────────────────────────╮    │
│ │ Epochs: 3                                                        │    │
│ │ Batch Size: 8                                                    │    │
│ │ Gradient Accumulation: 16                                        │    │
│ │ Learning Rate: 2e-5                                              │    │
│ │ Warmup Steps: 100                                                │    │
│ ╰──────────────────────────────────────────────────────────────────╯    │
│                                                                         │
│ ╭────────────────────── Distributed Config ───────────────────────╮    │
│ │ Strategy: FSDP                                                   │    │
│ │ World Size: 8                                                    │    │
│ │ Device: cuda                                                     │    │
│ │ Mixed Precision: bf16                                            │    │
│ ╰──────────────────────────────────────────────────────────────────╯    │
╰─────────────────────────────────────────────────────────────────────────╯
```

## ModelWatchCallback

Tracks model parameters and gradients with WandB:

::: dream_trainer.callbacks.ModelWatchCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - post_setup

### Usage

```python
from dream_trainer.callbacks import ModelWatchCallback

# Watch all gradients and parameters
watch = ModelWatchCallback(
    log="all",      # Log gradients and parameters
    log_freq=100    # Log every 100 batches
)

# Watch only gradients
watch = ModelWatchCallback(
    log="gradients",
    log_freq=50
)

# Watch only parameters
watch = ModelWatchCallback(
    log="parameters",
    log_freq=200
)
```

## Creating Custom Monitoring Callbacks

### Training Metrics Monitor

```python
class TrainingMonitor(Callback[BaseTrainer]):
    """Monitor and log detailed training metrics."""
    
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.gradient_norms = []
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        # Collect metrics
        self.losses.append(result["loss"].item())
        self.learning_rates.append(
            self.trainer.optimizer.param_groups[0]["lr"]
        )
        
        # Log every 100 steps
        if batch_idx % 100 == 0:
            self.log_summary()
    
    def log_summary(self):
        # Compute statistics
        recent_losses = self.losses[-100:]
        avg_loss = sum(recent_losses) / len(recent_losses)
        loss_std = torch.tensor(recent_losses).std().item()
        
        print(f"Average loss (last 100): {avg_loss:.4f} ± {loss_std:.4f}")
        print(f"Current LR: {self.learning_rates[-1]:.2e}")
```

### Memory Monitor

```python
class MemoryMonitor(Callback[BaseTrainer]):
    """Monitor GPU memory usage."""
    
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.peak_memory = 0
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        if batch_idx % self.log_every == 0:
            if torch.cuda.is_available():
                current = torch.cuda.memory_allocated() / 1024**3
                self.peak_memory = max(self.peak_memory, current)
                
                print(f"GPU Memory: {current:.2f} GB (peak: {self.peak_memory:.2f} GB)")
                
                # Log to trainer if available
                if hasattr(self.trainer, "log_scalar"):
                    self.trainer.log_scalar("system/gpu_memory_gb", current)
```

### Validation Improvement Monitor

```python
class ImprovementMonitor(RankZeroCallback[BaseTrainer]):
    """Track validation improvements."""
    
    def __init__(self, metric: str = "val_loss", mode: str = "min"):
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.epochs_without_improvement = 0
    
    def post_validation_epoch(self, result: dict[str, Any]):
        current_value = result.get(self.metric)
        if current_value is None:
            return
        
        # Check for improvement
        improved = False
        if self.mode == "min" and current_value < self.best_value:
            improved = True
            self.best_value = current_value
        elif self.mode == "max" and current_value > self.best_value:
            improved = True
            self.best_value = current_value
        
        if improved:
            print(f"✓ New best {self.metric}: {current_value:.4f}")
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            print(f"✗ No improvement for {self.epochs_without_improvement} epochs")
            print(f"  Best {self.metric}: {self.best_value:.4f}")
```

## Integration Examples

### Complete Monitoring Setup

```python
from dream_trainer.callbacks import (
    CallbackCollection,
    ProgressBar,
    LoggerCallback,
    ModelWatchCallback,
    TrainerSummary
)

# Create comprehensive monitoring
callbacks = CallbackCollection([
    # Display progress
    ProgressBar(metric="loss"),
    
    # Log metrics
    LoggerCallback(
        log_every_n_train_batches=10,
        log_every_n_val_batches=1
    ),
    
    # Track model
    ModelWatchCallback(log="gradients", log_freq=100),
    
    # Show summary
    TrainerSummary(),
    
    # Custom monitors
    MemoryMonitor(log_every=50),
    ImprovementMonitor(metric="val_accuracy", mode="max"),
])

config = DreamTrainerConfig(
    # ... other config ...
    callbacks=callbacks
)
```

### Conditional Monitoring

```python
class ConditionalMonitor(Callback[BaseTrainer]):
    """Enable detailed monitoring only when needed."""
    
    def __init__(self):
        self.detailed_logging = False
        self.loss_threshold = 1.0
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        loss = result["loss"]
        
        # Enable detailed logging if loss spikes
        if loss > self.loss_threshold:
            self.detailed_logging = True
            print(f"⚠️ Loss spike detected: {loss:.4f}")
            
            # Log additional information
            self.log_gradients()
            self.log_activations()
            self.log_weights()
    
    def log_gradients(self):
        for name, param in self.trainer.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 10:
                    print(f"Large gradient in {name}: {grad_norm:.2f}")
```

## Best Practices

### 1. Efficient Logging

```python
# Log frequently during debugging
debug_logger = LoggerCallback(
    log_every_n_train_batches=1,
    log_every_n_val_batches=1
)

# Log less frequently in production
prod_logger = LoggerCallback(
    log_every_n_train_batches=100,
    log_every_n_val_batches=10
)
```

### 2. Memory-Aware Monitoring

```python
class MemoryAwareLogger(MediaLoggerCallback):
    def post_validation_epoch(self, result: dict[str, Any]):
        # Only log media on specific epochs to save memory
        if self.trainer.current_epoch % 10 == 0:
            # Log limited samples
            samples = self.generate_samples(num=4)  # Not 100
            self.trainer.log_images(samples, desc="samples")
```

### 3. Distributed-Aware Monitoring

```python
class DistributedMonitor(RankZeroCallback[BaseTrainer]):
    """Only monitor on rank 0 to avoid duplicates."""
    
    def post_train_epoch(self, result: dict[str, Any]):
        # This only runs on rank 0
        print(f"Epoch {self.trainer.current_epoch} summary:")
        print(f"- Average loss: {result.get('loss', 'N/A')}")
        print(f"- Training time: {result.get('epoch_time', 'N/A')}s")
```

## See Also

- [Callback System](base.md) - Base callback documentation
- [Logger Mixins](../mixins/loggers.md) - Logging functionality
- [EvalMetricMixin](../mixins/eval_metric.md) - Metrics integration
- [WandB Integration](../mixins/loggers.md#wandbloggermixin) - WandB logging 