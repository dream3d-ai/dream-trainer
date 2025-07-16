# DreamTrainer

The `DreamTrainer` class is a production-ready trainer that demonstrates the full power of Dream Trainer's mixin architecture. It combines all major features into a single, cohesive training system.

## Overview

`DreamTrainer` combines:
- Base training functionality from `BaseTrainer`
- Model setup and parallelism from `SetupMixin`
- Evaluation metrics from `EvalMetricMixin`
- Weights & Biases logging from `WandBLoggerMixin`

This trainer serves as both a ready-to-use solution and a reference implementation for creating custom trainers.

## Class Reference

::: dream_trainer.trainer.DreamTrainer
    options:
      show_source: true
      show_bases: true
      show_root_full_path: false

## Configuration

::: dream_trainer.trainer.DreamTrainerConfig
    options:
      show_source: true
      show_bases: true

## Complete Example

Here's a full example using DreamTrainer for training a vision model:

```python
from dream_trainer.trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.configs import (
    DeviceParameters,
    TrainingParameters,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataLoaderConfig,
    LoggerConfig,
)
from dream_trainer.callbacks import CallbackCollection
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

class VisionTrainer(DreamTrainer):
    def configure_models(self):
        # Model is automatically tracked by SetupMixin
        self.model = torchvision.models.resnet50(
            pretrained=self.config.model_config.pretrained
        )
        self.model.fc = nn.Linear(2048, self.config.model_config.num_classes)
    
    def init_weights(self):
        # Initialize the new classifier head
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)
    
    def configure_optimizers(self):
        # Optimizer is automatically tracked
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer_config.learning_rate,
            weight_decay=self.config.optimizer_config.weight_decay
        )
    
    def configure_schedulers(self):
        # Scheduler is automatically tracked
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training_parameters.num_epochs
        )
    
    def configure_dataloaders(self):
        train_dataset = ImageDataset(split="train", transform=train_transforms)
        val_dataset = ImageDataset(split="val", transform=val_transforms)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.dataloader_config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.dataloader_config.val_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def configure_metrics(self):
        # Metrics are automatically tracked and moved to device
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.top5_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=10, top_k=5
        )
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=10)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Forward pass
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        
        # Backward with gradient accumulation
        with self.no_gradient_sync(self.model):
            self.backward(loss)
        
        # Step optimizer when not accumulating
        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.model, self.optimizer)
            
            # Log gradient norm
            self.log_scalar("train/grad_norm", grad_norm)
        
        # Log loss
        self.log_scalar("train/loss", loss)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Forward pass
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        
        # Update metrics
        self.accuracy(logits, labels)
        self.top5_accuracy(logits, labels)
        self.f1(logits, labels)
        
        # Log loss
        self.log_scalar("val/loss", loss)
        
        return {"val_loss": loss}

# Configure and run
config = DreamTrainerConfig(
    # Basic info
    seed=42,
    project="vision_classification",
    group="resnet50",
    experiment="baseline",
    
    # Device setup
    device_parameters=DeviceParameters(
        devices="cuda",
        num_nodes=1,
        strategy="fsdp",
        fsdp_config={"sharding_strategy": "FULL_SHARD"}
    ),
    
    # Training parameters
    training_parameters=TrainingParameters(
        num_epochs=90,
        gradient_accumulation_steps=4,
        gradient_clip_val=1.0,
        val_check_interval=1000,
        limit_val_batches=100
    ),
    
    # Model config
    model_config=ModelConfig(
        pretrained=True,
        num_classes=10,
        compile_mode="reduce-overhead"
    ),
    
    # Optimizer config
    optimizer_config=OptimizerConfig(
        learning_rate=1e-3,
        weight_decay=0.01
    ),
    
    # DataLoader config
    dataloader_config=DataLoaderConfig(
        train_batch_size=32,
        val_batch_size=64,
        num_workers=4
    ),
    
    # WandB config
    logger_config=LoggerConfig(
        enabled=True,
        log_every_n_steps=10,
        log_model="all"  # Log model at all checkpoints
    ),
    
    # Callbacks
    callbacks=CallbackCollection([
        # Add your callbacks here
    ])
)

# Create and run trainer
trainer = VisionTrainer(config)
trainer.configure()
trainer.setup()
trainer.fit()
```

## Mixin Features

### SetupMixin Features

The `SetupMixin` provides:

1. **Model Setup**
   - Automatic model registration via attributes
   - Weight initialization hooks
   - Parallelism strategies (FSDP, DDP, TP, PP)
   - Model compilation with torch.compile
   - Activation checkpointing

2. **Optimizer & Scheduler Setup**
   - Automatic optimizer registration
   - Learning rate scheduler management
   - Parameter group configuration

3. **DataLoader Setup**
   - Train and validation dataloader configuration
   - Automatic device placement
   - Distributed sampling

### EvalMetricMixin Features

The `EvalMetricMixin` provides:

1. **Metric Management**
   - Automatic metric registration via attributes
   - Device placement for metrics
   - Distributed synchronization
   - Metric computation and logging

2. **Integration with torchmetrics**
   - Support for all torchmetrics modules
   - Automatic reset between epochs
   - Proper aggregation across devices

### WandBLoggerMixin Features

The `WandBLoggerMixin` provides:

1. **Logging Capabilities**
   - Scalar logging with `log_scalar`
   - Image logging with `log_image`
   - Video logging with `log_video`
   - Histogram logging with `log_histogram`
   - Model watching and gradient tracking

2. **Experiment Tracking**
   - Automatic run initialization
   - Hyperparameter logging
   - System metrics tracking
   - Artifact management

## Creating Custom Trainers

To create your own trainer based on DreamTrainer:

1. **Choose Your Mixins**
   ```python
   class MyTrainer(BaseTrainer, SetupMixin, EvalMetricMixin):
       pass
   ```

2. **Match Configuration**
   ```python
   @dataclass
   class MyTrainerConfig(BaseTrainerConfig, SetupConfigMixin, EvalMetricConfigMixin):
       pass
   ```

3. **Implement Required Methods**
   - From SetupMixin: `configure_models`, `configure_optimizers`, `configure_dataloaders`
   - From EvalMetricMixin: `configure_metrics`
   - From BaseTrainer: `training_step`, `validation_step`

4. **Use Provided Features**
   - Call `self.backward()` for proper gradient scaling
   - Use `self.no_gradient_sync()` for gradient accumulation
   - Call `self.step()` for optimizer updates
   - Use `self.log_scalar()` for metrics logging

## See Also

- [BaseTrainer](base.md) - Core training functionality
- [SetupMixin](../mixins/setup.md) - Model and optimizer setup
- [EvalMetricMixin](../mixins/eval_metric.md) - Metrics integration
- [WandBLoggerMixin](../mixins/loggers.md#wandbloggermixin) - WandB logging 