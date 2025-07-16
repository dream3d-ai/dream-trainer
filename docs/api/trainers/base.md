# BaseTrainer

The `BaseTrainer` class provides a complete implementation of the training loop with support for gradient accumulation, validation, callbacks, and distributed training.

## Overview

`BaseTrainer` extends `AbstractTrainer` with:
- Complete training and validation loops
- Gradient accumulation and clipping
- Callback integration
- Mixed precision training support
- Distributed training synchronization
- Learning rate scheduling

## Class Reference

::: dream_trainer.trainer.BaseTrainer
    options:
      show_source: true
      show_bases: true
      show_root_full_path: false
      members:
        - __init__
        - state_dict
        - load_state_dict
        - fit
        - training_step
        - validation_step
        - eval
        - train
        - step
        - loss_parallel
        - backward
        - no_gradient_sync
        - total_gradient_norm
        - clip_gradient_norm
        - is_accumulating_gradients
        - train_context
        - perform_training_epoch
        - perform_validation_epoch
        - perform_sanity_validation_steps

## Configuration

::: dream_trainer.trainer.BaseTrainerConfig
    options:
      show_source: true
      show_bases: true

## Usage Example

```python
from dream_trainer.trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.configs import TrainingParameters, DeviceParameters
from dream_trainer.callbacks import CallbackCollection

class MyTrainer(BaseTrainer):
    def configure(self):
        # Define models
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Setup dataloaders
        self._train_dataloader = DataLoader(train_dataset, batch_size=32)
        self._val_dataloader = DataLoader(val_dataset, batch_size=32)
    
    def setup(self):
        # Move model to device
        self.model = self.model.to(self.world.device)
    
    def named_models(self):
        return {"main": self.model}
    
    def named_optimizers(self):
        return {"adam": self.optimizer}
    
    def named_schedulers(self):
        return None
    
    def get_module(self, fqn: str):
        return getattr(self.model, fqn)
    
    @property
    def train_dataloader(self):
        return self._train_dataloader
    
    @property
    def val_dataloader(self):
        return self._val_dataloader
    
    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        
        # Use no_gradient_sync for gradient accumulation
        with self.no_gradient_sync(self.model):
            self.backward(loss)
        
        # Step optimizer when not accumulating
        if not self.is_accumulating_gradients:
            self.step(self.model, self.optimizer)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return {"val_loss": loss}

# Create and run trainer
config = BaseTrainerConfig(
    seed=42,
    project="my_project",
    group="experiments",
    experiment="baseline",
    device_parameters=DeviceParameters(),
    training_parameters=TrainingParameters(
        num_epochs=10,
        gradient_accumulation_steps=4,
        gradient_clip_val=1.0,
        val_check_interval=100
    ),
    callbacks=CallbackCollection()
)

trainer = MyTrainer(config)
trainer.configure()
trainer.setup()
trainer.fit()
```

## Key Features

### Gradient Accumulation

BaseTrainer handles gradient accumulation automatically:

```python
# Configure in TrainingParameters
training_params = TrainingParameters(
    gradient_accumulation_steps=4,  # Accumulate over 4 batches
    gradient_clip_val=1.0
)

# In training_step, use no_gradient_sync
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    
    # Automatically handles gradient sync
    with self.no_gradient_sync(self.model):
        self.backward(loss)
    
    # Only step when not accumulating
    if not self.is_accumulating_gradients:
        self.step(self.model, self.optimizer)
```

### Mixed Precision Training

BaseTrainer integrates with PyTorch's autocast:

```python
def training_step(self, batch, batch_idx):
    # Forward pass is automatically in autocast context
    outputs = self.model(batch["input"])
    
    # Loss computation with loss parallelism
    with self.loss_parallel():
        loss = self.criterion(outputs, batch["target"])
    
    # Backward automatically handles mixed precision
    self.backward(loss)
```

### Validation

Validation runs automatically based on configuration:

```python
training_params = TrainingParameters(
    val_check_interval=100,  # Validate every 100 steps
    limit_val_batches=50,    # Use only 50 validation batches
    num_sanity_val_steps=2   # Run 2 sanity check steps
)
```

### Callbacks

BaseTrainer provides extensive callback hooks:

```python
# Callbacks are called at these points:
# - pre/post_train_epoch
# - pre/post_train_step
# - pre/post_validation_epoch
# - pre/post_validation_step
# - pre/post_optimizer_step
# - pre/post_optimizer_zero_grad
```

## Advanced Usage

### Custom Gradient Clipping

```python
def step(self, model, optimizer):
    # Compute gradient norm
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = self.total_gradient_norm(parameters)
    
    # Custom clipping logic
    if total_norm > self.training_parameters.gradient_clip_val:
        self.clip_gradient_norm(parameters, total_norm)
    
    # Continue with optimizer step
    optimizer.step()
```

### Distributed Training

BaseTrainer automatically handles distributed synchronization:

```python
# No gradient sync during accumulation
with self.no_gradient_sync(self.model):
    self.backward(loss)

# Automatic gradient sync on last accumulation step
if not self.is_accumulating_gradients:
    # Gradients are synchronized here
    self.step(self.model, self.optimizer)
```

## See Also

- [AbstractTrainer](abstract.md) - Base interface
- [DreamTrainer](dream.md) - Full-featured trainer
- [TrainingParameters](../configuration/training.md) - Training configuration
- [CallbackCollection](../callbacks/base.md) - Callback system 