# AbstractTrainer

The `AbstractTrainer` class is the foundation of the Dream Trainer framework. It defines the interface that all trainer implementations must follow.

## Overview

`AbstractTrainer` provides:
- Core training loop abstraction
- Distributed training world management
- Model, optimizer, and scheduler management
- State serialization interface
- Training lifecycle hooks

## Class Reference

::: dream_trainer.trainer.AbstractTrainer
    options:
      show_source: true
      show_bases: true
      show_root_full_path: false
      members:
        - __init__
        - named_models
        - named_optimizers
        - named_schedulers
        - get_module
        - train_dataloader
        - val_dataloader
        - state_dict
        - load_state_dict
        - fit
        - setup
        - configure
        - get_name_by_model
        - get_name_by_optimizer
        - get_name_by_scheduler
        - get_scheduler_from_optimizer

## Configuration

::: dream_trainer.trainer.AbstractTrainerConfig
    options:
      show_source: true
      show_bases: true

## Usage Example

```python
from dream_trainer.trainer import AbstractTrainer
from dream_trainer.configs import DeviceParameters

class MyTrainer(AbstractTrainer):
    def configure(self):
        # Define models, optimizers, etc.
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def setup(self):
        # Initialize components
        self.model = self.model.to(self.world.device)
    
    def named_models(self):
        return {"main": self.model}
    
    def named_optimizers(self):
        return {"adam": self.optimizer}
    
    def named_schedulers(self):
        return None  # No schedulers in this example
    
    def get_module(self, fqn: str):
        # Implement module lookup
        parts = fqn.split(".")
        module = self.named_models()[parts[0]]
        for part in parts[1:]:
            module = getattr(module, part)
        return module
    
    @property
    def train_dataloader(self):
        return self._train_dataloader
    
    @property
    def val_dataloader(self):
        return self._val_dataloader
    
    def state_dict(self):
        return {
            "models": {name: model.state_dict() for name, model in self.named_models().items()},
            "optimizers": {name: opt.state_dict() for name, opt in self.named_optimizers().items()},
            "trainer": {"global_step": self.global_step, "current_epoch": self.current_epoch}
        }
    
    def load_state_dict(self, state_dict):
        for name, model in self.named_models().items():
            model.load_state_dict(state_dict["models"][name])
        for name, opt in self.named_optimizers().items():
            opt.load_state_dict(state_dict["optimizers"][name])
        self.global_step = state_dict["trainer"]["global_step"]
        self.current_epoch = state_dict["trainer"]["current_epoch"]
    
    def fit(self):
        # Implement training loop
        pass
```

## Key Concepts

### World Management

The `world` attribute provides access to distributed training utilities:

```python
# Access device
device = trainer.world.device

# Check if distributed
if trainer.world.size > 1:
    print(f"Running on {trainer.world.size} processes")

# Get rank
rank = trainer.world.rank
```

### Named Components

The trainer uses a naming system for all components:

```python
# Access models by name
models = trainer.named_models()  # {"encoder": encoder, "decoder": decoder}

# Access optimizers by name
optimizers = trainer.named_optimizers()  # {"adam": adam_opt}

# Access schedulers by name
schedulers = trainer.named_schedulers()  # {"cosine": cosine_scheduler}
```

### State Management

The trainer tracks training progress:

```python
# Global step (optimizer steps)
print(f"Step: {trainer.global_step}")

# Current epoch
print(f"Epoch: {trainer.current_epoch}")

# Local batches (since program start)
print(f"Batches: {trainer.local_batches}")
```

## See Also

- [BaseTrainer](base.md) - Default implementation with common functionality
- [DreamTrainer](dream.md) - Production-ready trainer with all features
- [World Management](../utilities/world.md) - Distributed training utilities 