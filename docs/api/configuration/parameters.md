# Configuration Parameters

Dream Trainer uses strongly-typed configuration classes to manage all aspects of training. This ensures type safety, enables IDE autocomplete, and provides clear documentation of all available options.

## Training Configuration

### TrainingParameters

Core training hyperparameters:

::: dream_trainer.configs.TrainingParameters
    options:
      show_source: true
      show_bases: false

#### Usage Example

```python
from dream_trainer.configs import TrainingParameters

training_params = TrainingParameters(
    n_epochs=10,
    train_batch_size=32,
    gradient_clip_val=1.0,
    val_frequency=0.5,  # Validate every half epoch
    num_sanity_val_steps=2
)
```

## Device Configuration  

### DeviceParameters

Configuration for devices and parallelism strategies:

::: dream_trainer.configs.DeviceParameters
    options:
      show_source: true
      show_bases: false
      members:
        - __post_init__
        - validate
        - dp_shard
        - dp_replicate
        - tensor_parallel
        - context_parallel
        - pipeline_parallel
        - FSDP
        - HSDP
        - DDP
        - SINGLE_DEVICE

#### Usage Examples

##### Single Device
```python
from dream_trainer.configs import DeviceParameters

# Single GPU training
device_params = DeviceParameters.SINGLE_DEVICE(
    compile_model=True,
    checkpoint_activations=False
)
```

##### FSDP (Fully Sharded Data Parallel)
```python
# Automatic configuration
device_params = DeviceParameters.FSDP()

# Manual configuration
device_params = DeviceParameters.FSDP(
    tensor_parallel=8,    # TP within nodes
    dp_shard=4,          # FSDP across nodes
    cpu_offload=False,
    checkpoint_activations=True
)
```

##### HSDP (Hybrid Sharded Data Parallel)
```python
# HSDP with replication and sharding
device_params = DeviceParameters.HSDP(
    dp_shard=4,          # Shard within replicas
    tensor_parallel=8,   # TP degree
    compile_model=True
)
```

##### DDP (Distributed Data Parallel)
```python
# Traditional data parallelism
device_params = DeviceParameters.DDP(
    compile_model=True,
    checkpoint_activations=False
)
```

### Communication Configuration

::: dream_trainer.configs.Comm
    options:
      show_source: true
      show_bases: false

## Checkpoint Configuration

### CheckpointParameters

Configuration for model checkpointing:

::: dream_trainer.configs.CheckpointParameters
    options:
      show_source: true
      show_bases: false

#### Usage Example

```python
from dream_trainer.configs import CheckpointParameters

checkpoint_params = CheckpointParameters(
    enable=True,
    root_dir="./checkpoints",
    monitor="val_loss",
    resume_mode="min",  # Resume from best (minimum) loss
    checkpoint_every_n_epochs=1,
    keep_top_k=3,
    model_weights_only=False,
    exclude_from_loading=[],
    strict_load=True
)
```

## Logging Configuration

### LoggingParameters

Base logging configuration:

::: dream_trainer.configs.LoggingParameters
    options:
      show_source: true
      show_bases: false

### WandbLoggingParameters

Weights & Biases specific configuration:

::: dream_trainer.configs.WandbLoggingParameters
    options:
      show_source: true
      show_bases: false

## Fault Tolerance Configuration

### FaultToleranceParameters

Configuration for fault-tolerant training:

::: dream_trainer.configs.FaultToleranceParameters
    options:
      show_source: true
      show_bases: false

#### Usage Example

```python
from dream_trainer.configs import FaultToleranceParameters

ft_params = FaultToleranceParameters(
    enable=True,
    lighthouse_address="tcp://lighthouse:8080",
    min_replica_size=2,
    max_consecutive_failures=3
)
```

## Complete Configuration Example

Here's how to combine all configuration parameters:

```python
from dream_trainer.trainer import DreamTrainerConfig
from dream_trainer.configs import (
    TrainingParameters,
    DeviceParameters,
    CheckpointParameters,
    LoggingParameters,
)
from dream_trainer.callbacks import CallbackCollection

# Create complete configuration
config = DreamTrainerConfig(
    # Basic info
    seed=42,
    project="my_project",
    group="experiments",
    experiment="baseline_v1",
    
    # Device setup for multi-GPU
    device_parameters=DeviceParameters.FSDP(
        tensor_parallel=4,
        dp_shard="auto",
        compile_model=True,
        checkpoint_activations=True
    ),
    
    # Training hyperparameters
    training_parameters=TrainingParameters(
        n_epochs=100,
        train_batch_size=16,
        gradient_clip_val=1.0,
        val_frequency=0.2,  # 5 times per epoch
        num_sanity_val_steps=2
    ),
    
    # Model configuration
    model_config=ModelConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1
    ),
    
    # Optimizer configuration
    optimizer_config=OptimizerConfig(
        learning_rate=5e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    ),
    
    # Scheduler configuration
    scheduler_config=SchedulerConfig(
        warmup_steps=1000,
        scheduler_type="cosine"
    ),
    
    # DataLoader configuration
    dataloader_config=DataLoaderConfig(
        train_batch_size=16,
        val_batch_size=32,
        num_workers=4,
        prefetch_factor=2
    ),
    
    # Logging configuration
    logger_config=LoggingParameters(
        enabled=True
    ),
    
    # Callbacks
    callbacks=CallbackCollection([
        # Add your callbacks here
    ])
)
```

## Configuration Best Practices

### 1. Use Factory Methods

```python
# Good - use factory methods for common patterns
device_params = DeviceParameters.FSDP()

# Avoid - manual configuration is error-prone
device_params = DeviceParameters(
    _dp_shard=4,
    _dp_replicate=1,
    _tensor_parallel=8,
    # ... many more parameters
)
```

### 2. Validate Early

```python
# Configurations validate themselves
try:
    device_params = DeviceParameters(
        _dp_shard=3,
        _tensor_parallel=5
    )
    device_params.validate()  # Will raise if invalid
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

### 3. Environment-Based Configuration

```python
import os

# Adjust based on environment
if os.getenv("DISTRIBUTED_TRAINING"):
    device_params = DeviceParameters.FSDP()
else:
    device_params = DeviceParameters.SINGLE_DEVICE()
```

### 4. Configuration Inheritance

```python
from dataclasses import dataclass, field

@dataclass
class MyModelConfig:
    """Custom model configuration."""
    hidden_size: int = 768
    num_layers: int = 12
    vocab_size: int = 50257
    
@dataclass
class MyTrainerConfig(DreamTrainerConfig):
    """Extended trainer configuration."""
    model_config: MyModelConfig = field(default_factory=MyModelConfig)
    custom_param: float = 0.1
```

## Configuration Validation

Dream Trainer validates configurations at multiple levels:

1. **Type Checking**: Dataclasses ensure type correctness
2. **Value Validation**: `__post_init__` methods check constraints
3. **Runtime Validation**: `validate()` methods check compatibility

Example validation:

```python
# This will raise an error
checkpoint_params = CheckpointParameters(
    enable=True,
    root_dir="./checkpoints",
    checkpoint_every_n_epochs=None,  # Error: must specify frequency
    checkpoint_every_n_steps=None
)

# This will also raise an error
device_params = DeviceParameters(
    _dp_shard="auto",
    _tensor_parallel="auto"  # Error: only one can be "auto"
)
```

## See Also

- [Getting Started](../../getting-started.md) - Configuration examples
- [Device Configuration](device.md) - Detailed parallelism options
- [Training Configuration](training.md) - Training hyperparameters
- [Trainer Guide](../../trainer-guide.md) - Using configurations 