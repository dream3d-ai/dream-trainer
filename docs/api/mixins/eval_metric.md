# Evaluation Metrics Mixin

The `EvalMetricMixin` provides seamless integration with [torchmetrics](https://torchmetrics.readthedocs.io/) for standardized evaluation during training. It automatically handles metric device placement, distributed synchronization, and lifecycle management.

## Overview

The `EvalMetricMixin` enables:
- Automatic metric registration and tracking
- Device placement for GPU/TPU training
- Distributed metric synchronization
- Integration with any torchmetrics metric
- Clean separation of metric definition and usage

!!! note "Installation Required"
    This mixin requires torchmetrics to be installed:
    ```bash
    pip install dream-trainer[metrics]
    ```

## Class Reference

::: dream_trainer.trainer.mixins.EvalMetricMixin
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - configure_metrics
        - named_metrics
        - get_metric

### Configuration

::: dream_trainer.trainer.mixins.EvalMetricConfigMixin
    options:
      show_source: true
      show_bases: false

## Usage Example

```python
from dream_trainer.trainer.mixins import EvalMetricMixin, EvalMetricConfigMixin
from dream_trainer.trainer import BaseTrainer
import torchmetrics

class MyTrainer(BaseTrainer, EvalMetricMixin):
    def configure_metrics(self):
        # Metrics are automatically tracked when assigned as attributes
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=10
        )
        
        self.top5_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=10,
            top_k=5
        )
        
        self.f1_score = torchmetrics.F1Score(
            task="multiclass",
            num_classes=10,
            average="macro"
        )
        
        # MetricCollection for grouped metrics
        self.classification_metrics = torchmetrics.MetricCollection({
            "precision": torchmetrics.Precision(task="multiclass", num_classes=10),
            "recall": torchmetrics.Recall(task="multiclass", num_classes=10),
            "auroc": torchmetrics.AUROC(task="multiclass", num_classes=10)
        })
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        # Update metrics during training
        self.accuracy(logits, y)
        self.log_scalar("train/accuracy", self.accuracy)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        # Update all metrics
        self.accuracy(logits, y)
        self.top5_accuracy(logits, y)
        self.f1_score(logits, y)
        self.classification_metrics(logits, y)
        
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        # Compute and log final metrics
        self.log_scalar("val/accuracy", self.accuracy.compute())
        self.log_scalar("val/top5_accuracy", self.top5_accuracy.compute())
        self.log_scalar("val/f1_score", self.f1_score.compute())
        
        # Log collection metrics
        metrics = self.classification_metrics.compute()
        for name, value in metrics.items():
            self.log_scalar(f"val/{name}", value)
        
        # Reset metrics for next epoch
        self.accuracy.reset()
        self.top5_accuracy.reset()
        self.f1_score.reset()
        self.classification_metrics.reset()
```

## Common Metrics

### Classification Metrics

```python
def configure_metrics(self):
    # Accuracy variants
    self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    self.balanced_accuracy = torchmetrics.Accuracy(
        task="multiclass", 
        num_classes=10, 
        average="macro"
    )
    
    # Precision, Recall, F1
    self.precision = torchmetrics.Precision(
        task="multiclass", 
        num_classes=10,
        average="weighted"
    )
    self.recall = torchmetrics.Recall(
        task="multiclass",
        num_classes=10,
        average="weighted"
    )
    self.f1 = torchmetrics.F1Score(
        task="multiclass",
        num_classes=10,
        average="weighted"
    )
    
    # ROC and PR curves
    self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=10)
    self.avg_precision = torchmetrics.AveragePrecision(
        task="multiclass",
        num_classes=10
    )
```

### Regression Metrics

```python
def configure_metrics(self):
    # Error metrics
    self.mae = torchmetrics.MeanAbsoluteError()
    self.mse = torchmetrics.MeanSquaredError()
    self.rmse = torchmetrics.MeanSquaredError(squared=False)
    
    # Correlation metrics
    self.pearson = torchmetrics.PearsonCorrCoef()
    self.spearman = torchmetrics.SpearmanCorrCoef()
    
    # R-squared
    self.r2 = torchmetrics.R2Score()
```

### NLP Metrics

```python
def configure_metrics(self):
    # BLEU score for translation
    self.bleu = torchmetrics.BLEUScore()
    
    # ROUGE for summarization
    self.rouge = torchmetrics.ROUGEScore()
    
    # Perplexity for language modeling
    self.perplexity = torchmetrics.Perplexity()
    
    # Character/Word error rate
    self.cer = torchmetrics.CharErrorRate()
    self.wer = torchmetrics.WordErrorRate()
```

### Computer Vision Metrics

```python
def configure_metrics(self):
    # Segmentation metrics
    self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=20)
    self.dice = torchmetrics.Dice(multiclass=True, num_classes=20)
    
    # Detection metrics
    self.map = torchmetrics.detection.MeanAveragePrecision()
    
    # Image quality metrics
    self.psnr = torchmetrics.PeakSignalNoiseRatio()
    self.ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    self.fid = torchmetrics.image.FrechetInceptionDistance()
```

## MetricCollection

Use `MetricCollection` to group related metrics:

```python
def configure_metrics(self):
    # Group metrics by task
    self.train_metrics = torchmetrics.MetricCollection({
        "loss": torchmetrics.MeanMetric(),
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
        "learning_rate": torchmetrics.MeanMetric()
    })
    
    self.val_metrics = torchmetrics.MetricCollection({
        "loss": torchmetrics.MeanMetric(),
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=10),
        "auroc": torchmetrics.AUROC(task="multiclass", num_classes=10)
    })
    
    # Group by metric type
    self.classification_metrics = torchmetrics.MetricCollection({
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
        "precision": torchmetrics.Precision(task="multiclass", num_classes=10),
        "recall": torchmetrics.Recall(task="multiclass", num_classes=10),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=10)
    })
```

## Distributed Training

Metrics are automatically synchronized across devices in distributed training:

```python
def validation_step(self, batch, batch_idx):
    # Metrics automatically sync across GPUs
    logits = self.model(batch["input"])
    
    # Update happens on each device
    self.accuracy(logits, batch["target"])
    
    # Compute aggregates across all devices
    if batch_idx == self.num_val_batches - 1:
        # This automatically reduces across all processes
        final_accuracy = self.accuracy.compute()
        self.log_scalar("val/accuracy", final_accuracy)
```

## Custom Metrics

Create custom metrics by extending `torchmetrics.Metric`:

```python
class CustomMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        # Define state variables
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Update state based on predictions
        self.correct += (preds == target).sum()
        self.total += target.numel()
    
    def compute(self):
        # Compute final metric
        return self.correct.float() / self.total

# Use in trainer
def configure_metrics(self):
    self.custom_metric = CustomMetric()
```

## Best Practices

### 1. Metric Lifecycle

```python
def on_train_epoch_start(self):
    # Reset metrics at epoch start
    self.train_metrics.reset()

def training_step(self, batch, batch_idx):
    # Update metrics during training
    loss = self.compute_loss(batch)
    self.train_metrics["loss"](loss)
    
def on_train_epoch_end(self):
    # Compute and log final values
    metrics = self.train_metrics.compute()
    for name, value in metrics.items():
        self.log_scalar(f"train/{name}", value)
```

### 2. Memory Management

```python
def configure_metrics(self):
    # For large-scale metrics, compute on CPU
    self.confusion_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass",
        num_classes=1000,
        compute_on_cpu=True  # Avoid GPU OOM
    )
```

### 3. Metric Scheduling

```python
def validation_step(self, batch, batch_idx):
    # Only compute expensive metrics every N batches
    if batch_idx % 10 == 0:
        self.expensive_metric(preds, targets)
    
    # Always compute cheap metrics
    self.accuracy(preds, targets)
```

### 4. Multi-Task Metrics

```python
def configure_metrics(self):
    # Separate metrics for each task
    self.task_metrics = nn.ModuleDict({
        "classification": torchmetrics.Accuracy(task="multiclass", num_classes=10),
        "regression": torchmetrics.MeanSquaredError(),
        "segmentation": torchmetrics.JaccardIndex(task="multiclass", num_classes=20)
    })

def validation_step(self, batch, batch_idx):
    outputs = self.model(batch["input"])
    
    # Update task-specific metrics
    self.task_metrics["classification"](
        outputs["class_logits"], 
        batch["class_labels"]
    )
    self.task_metrics["regression"](
        outputs["regression"], 
        batch["regression_targets"]
    )
```

## Integration with Logging

The EvalMetricMixin works seamlessly with logger mixins:

```python
class MyTrainer(BaseTrainer, EvalMetricMixin, WandBLoggerMixin):
    def on_validation_epoch_end(self):
        # Compute all metrics
        metrics = {
            "accuracy": self.accuracy.compute(),
            "f1": self.f1_score.compute(),
            "confusion_matrix": self.confusion_matrix.compute()
        }
        
        # Log scalars
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log_scalar(f"val/{name}", value)
        
        # Log confusion matrix as image
        self.log_image(
            self.plot_confusion_matrix(metrics["confusion_matrix"]),
            desc="val/confusion_matrix"
        )
```

## See Also

- [torchmetrics documentation](https://torchmetrics.readthedocs.io/)
- [Logger Mixins](loggers.md) - For logging metric values
- [BaseTrainer](../trainers/base.md) - Core training functionality
- [DreamTrainer](../trainers/dream.md) - Complete example with metrics 