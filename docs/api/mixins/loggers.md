# Logger Mixins

The logger mixins provide standardized interfaces for experiment tracking and metric logging. Dream Trainer includes both a base `LoggerMixin` interface and specific implementations like `WandBLoggerMixin` for Weights & Biases integration.

## Overview

Logger mixins enable:

- Experiment tracking and configuration logging
- Scalar metric logging (loss, accuracy, etc.)
- Media logging (images, videos, plots)
- Model artifact tracking
- Distributed training support (only logs from rank 0)

## Base Logger Interface

### LoggerMixin

The base logger interface that all logging implementations should extend:

::: dream_trainer.trainer.mixins.LoggerMixin
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - log_config
        - log_dict
        - log_images
        - log_videos
        - log_plot

### Configuration

::: dream_trainer.trainer.mixins.LoggerConfigMixin
    options:
      show_source: true
      show_bases: false

## WandB Logger

The `WandBLoggerMixin` provides integration with [Weights & Biases](https://wandb.ai/) for comprehensive experiment tracking.

!!! note "Installation Required"
    This mixin requires wandb to be installed:
    ```bash
    pip install dream-trainer[wandb]
    ```

### WandBLoggerMixin

::: dream_trainer.trainer.mixins.WandBLoggerMixin
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - log_model
        - log_dict
        - log_scalar
        - log_images
        - log_videos
        - log_histogram
        - log_table
        - log_plot
        - log_config

### Configuration

::: dream_trainer.trainer.mixins.WandBLoggerConfigMixin
    options:
      show_source: true
      show_bases: false

## Usage Examples

### Basic Logging

```python
from dream_trainer.trainer.mixins import WandBLoggerMixin, WandBLoggerConfigMixin
from dream_trainer.trainer import BaseTrainer

class MyTrainer(BaseTrainer, WandBLoggerMixin):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        
        # Log scalar values
        self.log_scalar("train/loss", loss)
        self.log_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"])
        
        # Log multiple values at once
        self.log_dict({
            "train/batch_loss": loss,
            "train/grad_norm": grad_norm,
            "train/throughput": samples_per_sec
        })
        
        return {"loss": loss}
```

### Image Logging

```python
def validation_step(self, batch, batch_idx):
    images, labels = batch
    predictions = self.model(images)
    
    # Log a batch of images with predictions
    if batch_idx == 0:  # Log first batch only
        # Create visualization
        viz_images = self.create_prediction_viz(images, predictions, labels)
        
        self.log_images(
            viz_images,  # (B, C, H, W) tensor
            caption=[f"Pred: {p}, True: {l}" for p, l in zip(predictions, labels)],
            desc="val/predictions"
        )
    
    # Log attention maps
    if hasattr(self.model, "get_attention_maps"):
        attention_maps = self.model.get_attention_maps()
        self.log_images(
            attention_maps,
            desc="val/attention_maps"
        )
```

### Video Logging

```python
def on_validation_epoch_end(self):
    # Generate video samples
    generated_videos = self.model.generate(num_samples=4)  # (B, C, T, H, W)
    
    self.log_videos(
        generated_videos,
        caption=["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
        desc="val/generated_videos",
        fps=30
    )
```

### Histogram Logging

```python
def on_train_epoch_end(self):
    # Log weight distributions
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            self.log_histogram(
                param.data,
                desc=f"weights/{name}"
            )
            
            if param.grad is not None:
                self.log_histogram(
                    param.grad,
                    desc=f"gradients/{name}"
                )
```

### Table Logging

```python
def on_validation_epoch_end(self):
    # Create a table of predictions
    columns = ["image", "prediction", "ground_truth", "confidence"]
    data = []
    
    for batch in self.val_dataloader:
        images, labels = batch
        predictions = self.model(images)
        confidences = predictions.softmax(dim=-1).max(dim=-1).values
        
        for img, pred, label, conf in zip(images, predictions, labels, confidences):
            data.append([
                wandb.Image(img),
                self.class_names[pred.argmax()],
                self.class_names[label],
                conf.item()
            ])
        
        if len(data) >= 100:  # Limit table size
            break
    
    self.log_table(
        columns=columns,
        data=data,
        desc="val/predictions_table"
    )
```

### Plot Logging

```python
def create_loss_curve(self):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.plot(self.train_losses, label="Train")
    ax.plot(self.val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    
    return fig

def on_train_end(self):
    # Log matplotlib figure
    loss_curve = self.create_loss_curve()
    self.log_plot(loss_curve, desc="training/loss_curve")
    
    # Log confusion matrix
    cm_figure = self.plot_confusion_matrix(self.confusion_matrix)
    self.log_plot(cm_figure, desc="val/confusion_matrix")
```

## Model Tracking

WandB can track model weights and gradients:

```python
class MyTrainer(BaseTrainer, WandBLoggerMixin):
    def __init__(self, config):
        super().__init__(config)
        
        # Watch model for gradient and parameter tracking
        self.log_model(
            models=[self.model],  # List of models to track
            log="all",  # Log gradients and parameters
            log_freq=100  # Log every 100 batches
        )
```

## Configuration Logging

Log hyperparameters and configuration:

```python
def __init__(self, config):
    super().__init__(config)
    
    # Log full configuration
    self.log_config({
        "model": {
            "architecture": "transformer",
            "num_layers": self.config.num_layers,
            "hidden_dim": self.config.hidden_dim,
            "num_heads": self.config.num_heads
        },
        "training": {
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealing"
        },
        "data": {
            "dataset": self.config.dataset_name,
            "num_train_samples": len(self.train_dataset),
            "num_val_samples": len(self.val_dataset)
        }
    })
```

## Custom Logger Implementation

Create your own logger by extending `LoggerMixin`:

```python
from dream_trainer.trainer.mixins import LoggerMixin, LoggerConfigMixin
import tensorboardX

class TensorBoardLoggerMixin(LoggerMixin):
    def __init__(self, config):
        super().__init__(config)
        
        if self.world.is_global_zero and self.logging_parameters.enabled:
            self.writer = tensorboardX.SummaryWriter(
                log_dir=f"runs/{self.experiment}"
            )
        else:
            self.writer = None
    
    def log_dict(self, logs: dict[str, Any]):
        if self.writer is None:
            return
            
        for key, value in logs.items():
            if isinstance(value, (int, float, torch.Tensor)):
                self.writer.add_scalar(key, value, self.global_step)
    
    def log_images(self, images, caption=None, desc=None):
        if self.writer is None:
            return
            
        # Convert to grid
        grid = torchvision.utils.make_grid(images)
        self.writer.add_image(desc or "images", grid, self.global_step)
    
    def log_config(self, config):
        if self.writer is None:
            return
            
        # Log as text
        self.writer.add_text("config", str(config), 0)
```

## Best Practices

### 1. Conditional Logging

```python
def training_step(self, batch, batch_idx):
    # Only log every N steps to reduce overhead
    if self.global_step % self.logging_parameters.log_every_n_steps == 0:
        self.log_scalar("train/loss", loss)
        
    # Log expensive metrics less frequently
    if self.global_step % 100 == 0:
        self.log_histogram(self.model.embeddings.weight, "embeddings")
```

### 2. Distributed Training

The logger mixins automatically handle distributed training:

```python
# Only rank 0 logs to avoid duplicates
if self.world.is_global_zero:
    # This check is handled automatically by WandBLoggerMixin
    self.log_scalar("metric", value)
```

### 3. Memory Management

```python
def log_large_outputs(self, outputs):
    # Detach and move to CPU to avoid memory issues
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu()
    
    # Sample large outputs
    if len(outputs) > 100:
        indices = torch.randperm(len(outputs))[:100]
        outputs = outputs[indices]
    
    self.log_images(outputs, desc="samples")
```

### 4. Organizing Logs

Use hierarchical naming for better organization:

```python
# Use "/" to create sections in WandB
self.log_scalar("train/loss", train_loss)
self.log_scalar("train/accuracy", train_acc)
self.log_scalar("train/learning_rate", lr)

self.log_scalar("val/loss", val_loss)
self.log_scalar("val/accuracy", val_acc)

self.log_scalar("model/num_parameters", num_params)
self.log_scalar("model/grad_norm", grad_norm)
```

## Integration with Callbacks

Logger mixins work well with callbacks:

```python
class LoggingCallback(Callback):
    def on_train_batch_end(self, trainer, batch, batch_idx, outputs):
        # Access trainer's logging methods
        if hasattr(trainer, "log_scalar"):
            trainer.log_scalar("callback/batch_time", time.time() - self.start_time)
    
    def on_validation_epoch_end(self, trainer):
        # Log complex visualizations
        if hasattr(trainer, "log_plot"):
            fig = self.create_visualization(trainer)
            trainer.log_plot(fig, desc="callback/visualization")
```

## See Also

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [BaseTrainer](../trainers/base.md) - Core training functionality
- [EvalMetricMixin](eval_metric.md) - Metric computation
- [Callbacks](../callbacks/base.md) - Extending functionality 