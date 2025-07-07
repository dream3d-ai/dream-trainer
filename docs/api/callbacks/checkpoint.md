# Checkpoint Callbacks

Dream Trainer provides comprehensive checkpointing functionality through the `CheckpointCallback` class and its variants. These callbacks handle saving and restoring training state, enabling resumable training and model preservation.

## CheckpointCallback

The base checkpoint callback that handles synchronous checkpointing:

::: dream_trainer.callbacks.checkpoint.base.CheckpointCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - save
        - load

### Configuration

::: dream_trainer.configs.CheckpointParameters
    options:
      show_source: true
      show_bases: false

## AsyncCheckpointCallback

For improved performance with asynchronous saving:

::: dream_trainer.callbacks.checkpoint.async_checkpoint.AsyncCheckpointCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false

## Usage Examples

### Basic Checkpointing

```python
from dream_trainer.callbacks import CheckpointCallback
from dream_trainer.configs import CheckpointParameters

checkpoint_config = CheckpointParameters(
    root_dir="./checkpoints",
    monitor="val_loss",  # Metric to monitor
    mode="min",  # Save when metric is minimized
    keep_top_k=3,  # Keep best 3 checkpoints
    save_last=True,  # Always save last checkpoint
)

callback = CheckpointCallback(checkpoint_config)

# Add to trainer
config = DreamTrainerConfig(
    # ... other config ...
    callbacks=CallbackCollection([callback])
)
```

### Advanced Configuration

```python
checkpoint_config = CheckpointParameters(
    root_dir="/mnt/checkpoints",
    monitor="val_accuracy",
    mode="max",  # Save when metric is maximized
    keep_top_k=5,
    save_last=True,
    save_every_n_epochs=5,  # Save every 5 epochs
    save_every_n_steps=1000,  # Save every 1000 steps
    resume_mode="best",  # Resume from best checkpoint
)
```

### Custom Checkpoint Logic

```python
class CustomCheckpointCallback(CheckpointCallback):
    def should_save(self) -> bool:
        """Custom logic for when to save."""
        # Save only if validation loss improved by 1%
        if self._current_metric is None:
            return False
            
        if not hasattr(self, "best_metric"):
            self.best_metric = float("inf")
            
        improvement = (self.best_metric - self._current_metric) / self.best_metric
        if improvement > 0.01:
            self.best_metric = self._current_metric
            return True
            
        return False
```

## Checkpoint Structure

Checkpoints contain the complete training state:

```python
{
    "trainer": {
        "global_step": 10000,
        "current_epoch": 5,
        "callbacks": {...}  # Callback states
    },
    "models": {
        "model": {...},  # Model state dict
        "auxiliary": {...}  # Other models
    },
    "optimizers": {
        "optimizer": {...}  # Optimizer state
    },
    "schedulers": {
        "scheduler": {...}  # LR scheduler state
    },
    "dataloaders": {
        "train": {...},  # Dataloader state if stateful
        "val": {...}
    }
}
```

## Resuming Training

### Automatic Resume

```python
checkpoint_config = CheckpointParameters(
    root_dir="./checkpoints",
    resume_mode="latest",  # Resume from most recent
)

# Training will automatically resume if checkpoints exist
trainer = MyTrainer(config)
trainer.configure()
trainer.setup()
trainer.fit()  # Resumes from checkpoint
```

### Resume Modes

- `"latest"` - Resume from most recent checkpoint
- `"best"` - Resume from best checkpoint (by metric)
- `None` - Start fresh, ignore existing checkpoints

### Manual Loading

```python
# Load specific checkpoint
checkpoint_path = "checkpoints/experiment/checkpoint_step_10000"
state_dict = torch.load(checkpoint_path)
trainer.load_state_dict(state_dict)
```

## Monitoring Metrics

### Built-in Metrics

Common metrics to monitor:

```python
# Minimize these
monitor="loss"          # Training loss
monitor="val_loss"      # Validation loss
monitor="val_perplexity"  # Language model perplexity

# Maximize these
monitor="accuracy"      # Classification accuracy
monitor="val_f1"        # F1 score
monitor="val_bleu"      # Translation quality
```

### Custom Metrics

Return custom metrics from training/validation steps:

```python
def validation_step(self, batch, batch_idx):
    # Compute custom metric
    outputs = self.model(batch["input"])
    loss = self.criterion(outputs, batch["target"])
    
    # Custom metric calculation
    custom_score = self.compute_custom_metric(outputs, batch["target"])
    
    return {
        "val_loss": loss,
        "val_custom_score": custom_score,  # Can monitor this
    }
```

## Asynchronous Checkpointing

For large models, use asynchronous saving:

```python
from dream_trainer.callbacks import AsyncCheckpointCallback

# Same configuration, but saves happen in background
callback = AsyncCheckpointCallback(checkpoint_config)

# Training continues while checkpoint saves
```

Benefits:
- Non-blocking saves
- Better training throughput
- Automatic cleanup in background

## Best Practices

### 1. Choose Appropriate Metrics

```python
# For generative models
checkpoint_config = CheckpointParameters(
    monitor="val_perplexity",
    mode="min"
)

# For classification
checkpoint_config = CheckpointParameters(
    monitor="val_accuracy", 
    mode="max"
)

# For multi-task
checkpoint_config = CheckpointParameters(
    monitor="val_combined_score",  # Custom metric
    mode="max"
)
```

### 2. Storage Management

```python
checkpoint_config = CheckpointParameters(
    root_dir="/fast_ssd/checkpoints",  # Fast storage
    keep_top_k=3,  # Limit storage usage
    save_last=True,  # Keep recent for resume
)
```

### 3. Checkpoint Frequency

```python
# Balance between safety and performance
checkpoint_config = CheckpointParameters(
    save_every_n_steps=500,  # Frequent for unstable training
    save_every_n_epochs=1,   # Standard epoch checkpoints
    keep_top_k=5,            # Keep enough for analysis
)
```

### 4. Distributed Training

Checkpointing works seamlessly with distributed training:

```python
# Only rank 0 manages checkpoint metadata
# All ranks participate in saving model shards
# Automatic synchronization across ranks
```

## Checkpoint Utilities

### Finding Checkpoints

```python
from dream_trainer.callbacks.checkpoint.utils import find_checkpoints

# List all checkpoints
checkpoints = find_checkpoints(
    checkpoint_dir="./checkpoints/experiment",
    mode="best"  # or "latest"
)

for ckpt in checkpoints:
    print(f"Step: {ckpt.step}, Metric: {ckpt.metric}")
```

### Checkpoint Inspection

```python
# Load and inspect checkpoint
state_dict = torch.load("checkpoint_step_10000", map_location="cpu")

print(f"Global step: {state_dict['trainer']['global_step']}")
print(f"Epoch: {state_dict['trainer']['current_epoch']}")
print(f"Model keys: {state_dict['models'].keys()}")
```

## Integration with Other Callbacks

Checkpointing integrates with other callbacks:

```python
class MetricCheckpointCallback(Callback[BaseTrainer]):
    """Save checkpoint when multiple metrics improve."""
    
    def __init__(self, checkpoint_callback: CheckpointCallback):
        self.checkpoint_callback = checkpoint_callback
        self.best_accuracy = 0
        self.best_f1 = 0
    
    def post_validation_epoch(self, result: dict[str, Any]):
        accuracy = result.get("val_accuracy", 0)
        f1 = result.get("val_f1", 0)
        
        # Save if both metrics improve
        if accuracy > self.best_accuracy and f1 > self.best_f1:
            self.best_accuracy = accuracy
            self.best_f1 = f1
            
            # Trigger checkpoint save
            self.checkpoint_callback._current_metric = accuracy
            self.checkpoint_callback.save()
```

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Check paths and experiment names
2. **Resume fails**: Ensure model architecture matches
3. **Disk space**: Monitor available storage
4. **Slow saves**: Use async checkpointing for large models

### Debugging

```python
# Enable verbose logging
import logging
logging.getLogger("dream_trainer.callbacks.checkpoint").setLevel(logging.DEBUG)

# Check checkpoint contents
ckpt = torch.load("checkpoint", map_location="cpu")
print(f"Checkpoint keys: {ckpt.keys()}")
print(f"Model state dict size: {len(ckpt['models']['model'])}")
```

## See Also

- [Callback System](base.md) - Base callback documentation
- [Configuration](../configuration/parameters.md) - CheckpointParameters details
- [BaseTrainer](../trainers/base.md) - State dict implementation 