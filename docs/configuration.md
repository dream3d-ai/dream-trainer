# Configuration Guide

This guide explains all configuration options available in Dream Trainer.

## Table of Contents

- [Basic Configuration](#basic-configuration)
- [Device Parameters](#device-parameters)
- [Training Parameters](#training-parameters)
- [Checkpoint Parameters](#checkpoint-parameters)
- [Logging Parameters](#logging-parameters)
- [Advanced Configuration](#advanced-configuration)

## Basic Configuration

The main configuration class is `DreamTrainerConfig`. Here's a basic example:

```python
from dream_trainer import DreamTrainerConfig

config = DreamTrainerConfig(
    project="my-project",
    group="experiments",
    experiment="run-001"
)
```

### Project Settings

| Parameter    | Type | Description                        |
| ------------ | ---- | ---------------------------------- |
| `project`    | str  | Project name for organization      |
| `group`      | str  | Group name for related experiments |
| `experiment` | str  | Unique experiment identifier       |

## Device Parameters

Configure hardware and distributed training settings:

```python
from dream_trainer.configs import DeviceParameters
import torch

device_params = DeviceParameters(
    # Distributed training
    data_parallel_size=1,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    context_parallel_size=1,

    # Performance
    compile_model=True,
    param_dtype=torch.bfloat16,
    activation_dtype=torch.bfloat16,

    # Memory optimization
    checkpoint_activations=False,
    offload_optimizer=False,
    offload_parameters=False
)
```

### Distributed Training

| Parameter                | Type | Description                         |
| ------------------------ | ---- | ----------------------------------- |
| `data_parallel_size`     | int  | Number of GPUs for data parallelism |
| `tensor_parallel_size`   | int  | Tensor parallelism degree           |
| `pipeline_parallel_size` | int  | Pipeline parallelism degree         |
| `context_parallel_size`  | int  | Context parallelism degree          |

### Performance Settings

| Parameter          | Type        | Description                          |
| ------------------ | ----------- | ------------------------------------ |
| `compile_model`    | bool        | Use torch.compile for optimization   |
| `param_dtype`      | torch.dtype | Parameter data type (e.g., bfloat16) |
| `activation_dtype` | torch.dtype | Activation data type                 |

### Memory Optimization

| Parameter                | Type | Description                     |
| ------------------------ | ---- | ------------------------------- |
| `checkpoint_activations` | bool | Enable activation checkpointing |
| `offload_optimizer`      | bool | Offload optimizer states to CPU |
| `offload_parameters`     | bool | Offload parameters to CPU       |

## Training Parameters

Configure training loop settings:

```python
from dream_trainer.configs import TrainingParameters

training_params = TrainingParameters(
    # Basic training
    n_epochs=10,
    train_batch_size=32,
    val_batch_size=32,

    # Optimization
    gradient_clip_val=1.0,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,

    # Validation
    val_frequency=0.5,
    num_sanity_val_steps=2,

    # Learning rate
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000
)
```

### Basic Training

| Parameter          | Type | Description               |
| ------------------ | ---- | ------------------------- |
| `n_epochs`         | int  | Number of training epochs |
| `train_batch_size` | int  | Training batch size       |
| `val_batch_size`   | int  | Validation batch size     |

### Optimization

| Parameter                     | Type  | Description                     |
| ----------------------------- | ----- | ------------------------------- |
| `gradient_clip_val`           | float | Gradient clipping value         |
| `gradient_accumulation_steps` | int   | Steps for gradient accumulation |
| `max_grad_norm`               | float | Maximum gradient norm           |

### Validation

| Parameter              | Type  | Description                   |
| ---------------------- | ----- | ----------------------------- |
| `val_frequency`        | float | Validation frequency (epochs) |
| `num_sanity_val_steps` | int   | Sanity check steps            |

### Learning Rate

| Parameter       | Type  | Description                |
| --------------- | ----- | -------------------------- |
| `learning_rate` | float | Initial learning rate      |
| `weight_decay`  | float | Weight decay coefficient   |
| `warmup_steps`  | int   | Learning rate warmup steps |

## Checkpoint Parameters

Configure model checkpointing:

```python
from dream_trainer.configs import CheckpointParameters

checkpoint_params = CheckpointParameters(
    # Basic settings
    root_dir="./checkpoints",
    monitor="val_loss",
    mode="min",

    # Checkpoint frequency
    checkpoint_every_n_epochs=1,
    checkpoint_every_n_steps=None,

    # Checkpoint management
    keep_top_k=3,
    save_last=True,

    # Resume settings
    resume_mode="latest",  # or "best"
    resume_path=None
)
```

### Basic Settings

| Parameter  | Type | Description                         |
| ---------- | ---- | ----------------------------------- |
| `root_dir` | str  | Checkpoint directory                |
| `monitor`  | str  | Metric to monitor                   |
| `mode`     | str  | "min" or "max" for monitored metric |

### Checkpoint Frequency

| Parameter                   | Type | Description         |
| --------------------------- | ---- | ------------------- |
| `checkpoint_every_n_epochs` | int  | Save every N epochs |
| `checkpoint_every_n_steps`  | int  | Save every N steps  |

### Checkpoint Management

| Parameter    | Type | Description             |
| ------------ | ---- | ----------------------- |
| `keep_top_k` | int  | Keep best K checkpoints |
| `save_last`  | bool | Always save latest      |

### Resume Settings

| Parameter     | Type | Description        |
| ------------- | ---- | ------------------ |
| `resume_mode` | str  | "latest" or "best" |
| `resume_path` | str  | Path to checkpoint |

## Logging Parameters

Configure experiment tracking:

```python
from dream_trainer.configs import WandBParameters

wandb_params = WandBParameters(
    # Basic settings
    project="my-project",
    entity="my-team",

    # Experiment info
    tags=["experiment", "classification"],
    notes="Initial baseline run",

    # Logging settings
    log_model=True,
    log_artifacts=True,
    log_code=True
)
```

### Basic Settings

| Parameter | Type | Description        |
| --------- | ---- | ------------------ |
| `project` | str  | WandB project name |
| `entity`  | str  | WandB entity/team  |

### Experiment Info

| Parameter | Type      | Description      |
| --------- | --------- | ---------------- |
| `tags`    | List[str] | Experiment tags  |
| `notes`   | str       | Experiment notes |

### Logging Settings

| Parameter       | Type | Description           |
| --------------- | ---- | --------------------- |
| `log_model`     | bool | Log model checkpoints |
| `log_artifacts` | bool | Log artifacts         |
| `log_code`      | bool | Log code changes      |

## Advanced Configuration

### Custom Configuration

You can create custom configuration classes:

```python
from dream_trainer.configs import BaseConfig

class CustomConfig(BaseConfig):
    def __init__(
        self,
        custom_param: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_param = custom_param
```

### Configuration Inheritance

Configurations can be inherited and extended:

```python
class ExtendedConfig(DreamTrainerConfig):
    def __init__(
        self,
        new_param: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.new_param = new_param
```

### Configuration Validation

Add validation to your configurations:

```python
from dream_trainer.configs import BaseConfig
from typing import Optional

class ValidatedConfig(BaseConfig):
    def __init__(
        self,
        required_param: str,
        optional_param: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.required_param = required_param
        self.optional_param = optional_param

    def validate(self):
        if not self.required_param:
            raise ValueError("required_param cannot be empty")
        if self.optional_param is not None and self.optional_param < 0:
            raise ValueError("optional_param must be non-negative")
```

## Best Practices

1. **Use Type Hints**: Always use type hints for better IDE support
2. **Validate Inputs**: Add validation for critical parameters
3. **Document Parameters**: Add docstrings for custom parameters
4. **Use Sensible Defaults**: Provide reasonable default values
5. **Group Related Parameters**: Use nested configs for related settings

## Environment Variables

Dream Trainer respects several environment variables:

```bash
# PyTorch distributed settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0

# NCCL settings for better performance
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

# GPU memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Configuration Validation

Dream Trainer validates configurations at runtime:

```python
# These will raise errors:
DeviceParameters(
    data_parallel_size=3,
    tensor_parallel_size=2,
    # Error: Total devices (6) must match available GPUs
)

TrainingParameters(
    train_batch_size=7,
    # Error: Batch size must be divisible by data parallel size
)
```

## Next Steps

- See [Trainer Guide](trainer-guide.md) for implementing custom trainers
- Check [Callbacks](callbacks.md) for extending functionality
- Read [Distributed Training](distributed.md) for multi-GPU details
