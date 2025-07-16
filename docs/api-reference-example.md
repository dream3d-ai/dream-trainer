# Example API Reference: BaseTrainer

This is an example of what the API reference documentation should look like for Dream Trainer. This example shows the documentation for `BaseTrainer`.

---

# dream_trainer.trainer.BaseTrainer

```python
class dream_trainer.trainer.BaseTrainer(config: BaseTrainerConfig, *args, **kwargs)
```

An implementation of a basic training loop with support for gradient accumulation, validation, callbacks, and distributed training.

`BaseTrainer` provides the core training loop functionality and serves as the foundation for all trainers in Dream Trainer. It handles the training lifecycle, manages callbacks, and provides hooks for customization through abstract methods.

## Parameters

- **config** (*BaseTrainerConfig*) – Configuration object containing training parameters and callbacks
- ***args** – Additional positional arguments passed to parent class
- ****kwargs** – Additional keyword arguments passed to parent class

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `BaseTrainerConfig` | Configuration object for the trainer |
| `callbacks` | `CallbackCollection` | Collection of callbacks to execute during training |
| `training` | `bool` | Whether the trainer is currently in training mode |
| `global_step` | `int` | Number of optimizer steps taken across all epochs |
| `local_batches` | `int` | Number of batches processed since program start |
| `current_epoch` | `int` | Current epoch number (0-indexed) |

## Abstract Methods

These methods must be implemented by subclasses:

### training_step

```python
abstract training_step(batch: dict[str, Any], batch_idx: int) → dict[str, Any]
```

Execute a single training step.

This method should implement the forward pass, loss computation, and backward pass for a single batch of training data.

**Parameters:**
- **batch** (*dict[str, Any]*) – Dictionary containing the batch data, typically with keys like 'input', 'target', etc.
- **batch_idx** (*int*) – Index of the current batch within the epoch

**Returns:**
- *dict[str, Any]* – Dictionary containing at minimum the computed loss and any other metrics or values to log

**Example:**
```python
def training_step(self, batch, batch_idx):
    inputs, targets = batch['inputs'], batch['targets']
    outputs = self.model(inputs)
    loss = F.cross_entropy(outputs, targets)
    
    # Use self.backward() for proper gradient accumulation
    self.backward(loss)
    
    # Only step optimizer when not accumulating
    if not self.is_accumulating_gradients:
        grad_norm = self.step(self.model, self.optimizer)
        return {"loss": loss, "grad_norm": grad_norm}
    
    return {"loss": loss}
```

### validation_step

```python
abstract validation_step(batch: dict[str, Any], batch_idx: int) → dict[str, Any]
```

Execute a single validation step.

Similar to `training_step` but without gradient computation. Used for model evaluation during validation phases.

**Parameters:**
- **batch** (*dict[str, Any]*) – Dictionary containing the batch data
- **batch_idx** (*int*) – Index of the current batch

**Returns:**
- *dict[str, Any]* – Dictionary containing validation metrics

**Example:**
```python
def validation_step(self, batch, batch_idx):
    inputs, targets = batch['inputs'], batch['targets']
    
    with torch.no_grad():
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
    
    return {
        "val_loss": loss,
        "val_accuracy": accuracy
    }
```

## Core Methods

### fit

```python
fit() → None
```

Execute the complete training pipeline.

This is the main entry point for training. It handles the entire training lifecycle including setup, training loops, validation, and cleanup. The method ensures proper cleanup by destroying the distributed process group in the finally block, even if training is interrupted.

**Example:**
```python
trainer = MyTrainer(config)
trainer.fit()  # Runs the complete training pipeline
```

### backward

```python
backward(loss: torch.Tensor) → None
```

Perform backward pass with gradient accumulation support.

This method handles gradient scaling for mixed precision training and gradient accumulation. It should be used instead of `loss.backward()` directly.

**Parameters:**
- **loss** (*torch.Tensor*) – The loss tensor to backpropagate

**Example:**
```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.backward(loss)  # Handles accumulation and scaling
    return {"loss": loss}
```

### step

```python
step(model: nn.Module, optimizer: Optimizer, scheduler: Optional[LRScheduler] = None) → float
```

Perform optimizer step with gradient clipping and scheduler update.

**Parameters:**
- **model** (*nn.Module*) – Model to compute gradient norm for
- **optimizer** (*Optimizer*) – Optimizer to step
- **scheduler** (*Optional[LRScheduler]*) – Learning rate scheduler to step

**Returns:**
- *float* – The gradient norm before clipping

**Example:**
```python
if not self.is_accumulating_gradients:
    grad_norm = self.step(self.model, self.optimizer, self.scheduler)
    self.log("grad_norm", grad_norm)
```

## Properties

### is_accumulating_gradients

```python
property is_accumulating_gradients: bool
```

Whether the trainer is currently accumulating gradients.

Returns `True` if the current step is not the last step in a gradient accumulation cycle.

**Example:**
```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.backward(loss)
    
    if not self.is_accumulating_gradients:
        # Only log when we actually update weights
        self.log("learning_rate", self.optimizer.param_groups[0]["lr"])
    
    return {"loss": loss}
```

## Hooks and Callbacks

The trainer provides numerous hooks throughout the training lifecycle:

### Training Hooks
- `pre_launch` – Before distributed world launch
- `pre_configure` – Before configuration
- `post_configure` – After configuration
- `pre_setup` – Before setup
- `post_setup` – After setup
- `pre_fit` – Before training starts
- `pre_epoch` – Before each epoch
- `post_epoch` – After each epoch
- `post_fit` – After training completes

### Batch Hooks
- `pre_train_batch` – Before each training batch
- `post_train_batch` – After each training batch
- `pre_val_batch` – Before each validation batch
- `post_val_batch` – After each validation batch

## Complete Example

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from dream_trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin

@dataclass
class MyConfig(BaseTrainerConfig, SetupConfigMixin):
    learning_rate: float = 1e-3
    hidden_size: int = 128

class MyTrainer(BaseTrainer, SetupMixin):
    config: MyConfig
    
    def configure_models(self):
        """Create model on meta device"""
        self.model = nn.Sequential(
            nn.Linear(784, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 10)
        )
    
    def init_weights(self):
        """Initialize weights after device placement"""
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.model.apply(init_fn)
    
    def configure_optimizers(self):
        """Set up optimizer"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def configure_dataloaders(self):
        """Return train and validation dataloaders"""
        train_dataset = MyDataset("train")
        val_dataset = MyDataset("val")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_parameters.train_batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_parameters.val_batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x.view(x.size(0), -1))
        loss = F.cross_entropy(logits, y)
        
        self.backward(loss)
        
        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.model, self.optimizer)
            accuracy = (logits.argmax(dim=1) == y).float().mean()
            
            return {
                "loss": loss,
                "accuracy": accuracy,
                "grad_norm": grad_norm
            }
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        with torch.no_grad():
            logits = self.model(x.view(x.size(0), -1))
            loss = F.cross_entropy(logits, y)
            accuracy = (logits.argmax(dim=1) == y).float().mean()
        
        return {
            "val_loss": loss,
            "val_accuracy": accuracy
        }

# Usage
config = MyConfig(
    project="mnist-classification",
    group="baseline",
    experiment="run-001",
    training_parameters=TrainingParameters(
        n_epochs=10,
        train_batch_size=32,
        gradient_accumulation_steps=4
    )
)

trainer = MyTrainer(config)
trainer.fit()
```

## See Also

- [`AbstractTrainer`](abstract.md) – Base interface for all trainers
- [`DreamTrainer`](dream.md) – Full-featured trainer with all mixins
- [Training Guide](../guides/training.md) – Detailed training concepts
- [Configuration Guide](../guides/configuration.md) – Configuration patterns

## Notes

- The trainer automatically handles distributed training setup when launched with `torchrun`
- Gradient accumulation is handled transparently - just call `self.backward(loss)`
- Always use `self.step()` instead of calling `optimizer.step()` directly
- The trainer manages device placement - models are created on meta device and materialized during setup

---

This example demonstrates the level of detail and organization expected for API reference documentation. Each class should have similar comprehensive documentation with examples, parameter descriptions, and cross-references. 