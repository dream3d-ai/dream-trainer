# Trainer Guide

This guide explains how to create and customize trainers in Dream Trainer.

## Table of Contents
- [Basic Trainer](#basic-trainer)
- [Training Loop](#training-loop)
- [Model Configuration](#model-configuration)
- [Optimizer Configuration](#optimizer-configuration)
- [DataLoader Configuration](#dataloader-configuration)
- [Advanced Features](#advanced-features)

## Basic Trainer

The simplest way to create a trainer is to extend `DreamTrainer`:

```python
from dream_trainer import DreamTrainer, DreamTrainerConfig
import torch.nn as nn

class MyTrainer(DreamTrainer):
    def __init__(self, config: DreamTrainerConfig, model: nn.Module):
        super().__init__(config)
        self.model = model
```

### Required Methods

Every trainer must implement these methods:

```python
class MyTrainer(DreamTrainer):
    def configure_models(self):
        """Configure your model(s) here"""
        pass
        
    def configure_optimizers(self):
        """Configure optimizer(s)"""
        pass
        
    def configure_dataloaders(self):
        """Configure train and validation dataloaders"""
        pass
        
    def training_step(self, batch, batch_idx):
        """Define a single training step"""
        pass
        
    def validation_step(self, batch, batch_idx):
        """Define a single validation step"""
        pass
```

## Training Loop

### Training Step

The training step defines what happens in each iteration:

```python
def training_step(self, batch, batch_idx):
    # Unpack batch
    inputs, targets = batch
    
    # Forward pass
    outputs = self.model(inputs)
    loss = self.criterion(outputs, targets)
    
    # Backward pass (handled automatically)
    self.backward(loss)
    
    # Return metrics to log
    return {
        "loss": loss,
        "accuracy": self.compute_accuracy(outputs, targets)
    }
```

### Validation Step

The validation step is similar but without backward pass:

```python
def validation_step(self, batch, batch_idx):
    # Unpack batch
    inputs, targets = batch
    
    # Forward pass
    outputs = self.model(inputs)
    loss = self.criterion(outputs, targets)
    
    # Return metrics to log
    return {
        "val_loss": loss,
        "val_accuracy": self.compute_accuracy(outputs, targets)
    }
```

### Custom Training Loop

For more control, you can override the training loop:

```python
def training_loop(self):
    """Custom training loop"""
    for epoch in range(self.current_epoch, self.config.n_epochs):
        # Training
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            # Custom training logic
            outputs = self.training_step(batch, batch_idx)
            
            # Custom logging
            self.log_metrics(outputs)
            
        # Validation
        if self.should_validate():
            self.validation_loop()
```

## Model Configuration

### Single Model

Configure a single model:

```python
def configure_models(self):
    """Configure a single model"""
    # Model is automatically moved to device
    # and wrapped with distributed training wrappers
    pass
```

### Multiple Models

Configure multiple models (e.g., GAN):

```python
def configure_models(self):
    """Configure multiple models"""
    # Generator
    self.generator = self.generator.to(self.device)
    if self.is_distributed:
        self.generator = self.wrap_model(self.generator)
        
    # Discriminator
    self.discriminator = self.discriminator.to(self.device)
    if self.is_distributed:
        self.discriminator = self.wrap_model(self.discriminator)
```

### Model Compilation

Enable model compilation for better performance:

```python
def configure_models(self):
    """Configure model with compilation"""
    if self.config.compile_model:
        self.model = torch.compile(self.model)
```

## Optimizer Configuration

### Basic Optimizer

Configure a single optimizer:

```python
def configure_optimizers(self):
    """Configure a single optimizer"""
    self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.config.learning_rate,
        weight_decay=self.config.weight_decay
    )
```

### Multiple Optimizers

Configure multiple optimizers:

```python
def configure_optimizers(self):
    """Configure multiple optimizers"""
    # Generator optimizer
    self.g_optimizer = torch.optim.AdamW(
        self.generator.parameters(),
        lr=self.config.g_lr
    )
    
    # Discriminator optimizer
    self.d_optimizer = torch.optim.AdamW(
        self.discriminator.parameters(),
        lr=self.config.d_lr
    )
```

### Learning Rate Schedulers

Add learning rate schedulers:

```python
def configure_optimizers(self):
    """Configure optimizer with scheduler"""
    # Optimizer
    self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.config.learning_rate
    )
    
    # Scheduler
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=self.config.n_epochs
    )
```

## DataLoader Configuration

### Basic DataLoaders

Configure train and validation dataloaders:

```python
def configure_dataloaders(self):
    """Configure basic dataloaders"""
    # Training data
    train_dataset = MyDataset(
        data_dir=self.config.train_data_dir,
        split="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config.train_batch_size,
        shuffle=True,
        num_workers=self.config.num_workers
    )
    
    # Validation data
    val_dataset = MyDataset(
        data_dir=self.config.val_data_dir,
        split="val"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=self.config.val_batch_size,
        shuffle=False,
        num_workers=self.config.num_workers
    )
    
    return train_loader, val_loader
```

### Distributed DataLoaders

Configure distributed dataloaders:

```python
def configure_dataloaders(self):
    """Configure distributed dataloaders"""
    # Training data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=self.world_size,
        rank=self.global_rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config.train_batch_size,
        sampler=train_sampler,
        num_workers=self.config.num_workers
    )
    
    # Validation data
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=self.world_size,
        rank=self.global_rank,
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=self.config.val_batch_size,
        sampler=val_sampler,
        num_workers=self.config.num_workers
    )
    
    return train_loader, val_loader
```

## Advanced Features

### Custom Metrics

Add custom metrics:

```python
def compute_metrics(self, outputs, targets):
    """Compute custom metrics"""
    return {
        "accuracy": self.compute_accuracy(outputs, targets),
        "f1_score": self.compute_f1(outputs, targets),
        "precision": self.compute_precision(outputs, targets),
        "recall": self.compute_recall(outputs, targets)
    }
```

### Gradient Clipping

Enable gradient clipping:

```python
def training_step(self, batch, batch_idx):
    # ... training logic ...
    
    # Gradient clipping
    if self.config.gradient_clip_val is not None:
        self.clip_gradients(
            self.optimizer,
            max_norm=self.config.gradient_clip_val
        )
```

### Mixed Precision

Enable mixed precision training:

```python
def training_step(self, batch, batch_idx):
    # Automatic mixed precision
    with self.autocast():
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
    
    # Backward pass with scaling
    self.backward(loss)
```

### Checkpointing

Save and load checkpoints:

```python
def save_checkpoint(self, path):
    """Save checkpoint"""
    checkpoint = {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "epoch": self.current_epoch,
        "config": self.config
    }
    torch.save(checkpoint, path)

def load_checkpoint(self, path):
    """Load checkpoint"""
    checkpoint = torch.load(path)
    self.model.load_state_dict(checkpoint["model"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    self.current_epoch = checkpoint["epoch"]
```

### Custom Callbacks

Add custom callbacks:

```python
class MyCallback(Callback):
    def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
        # Custom logic after each training batch
        pass
        
    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx):
        # Custom logic after each validation batch
        pass
```

## Best Practices

1. **Keep It Simple**: Start with basic implementation, then add features
2. **Use Type Hints**: Add type hints for better IDE support
3. **Document Methods**: Add docstrings to explain functionality
4. **Handle Errors**: Add proper error handling and logging
5. **Test Thoroughly**: Write unit tests for your trainer
6. **Profile Performance**: Monitor memory usage and training speed
7. **Use Callbacks**: Extend functionality through callbacks
8. **Follow PyTorch**: Follow PyTorch best practices and patterns
