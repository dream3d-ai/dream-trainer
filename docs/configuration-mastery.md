# Configuration Mastery

This guide dives deep into advanced configuration patterns for Dream Trainer, showcasing how to leverage Python's type system for powerful, maintainable, and production-ready configurations.

## Table of Contents

- [Configuration Patterns](#configuration-patterns)
  - [Factory Functions](#factory-functions)
  - [Conditional Configuration](#conditional-configuration)
  - [Environment-Based Configs](#environment-based-configs)
  - [Multi-Experiment Management](#multi-experiment-management)
- [Validation & Constraints](#validation--constraints)
  - [Custom Validators](#custom-validators)
  - [Type Safety Patterns](#type-safety-patterns)
  - [Configuration Testing](#configuration-testing)
- [Real-World Examples](#real-world-examples)
  - [Multi-Node Configurations](#multi-node-configurations)
  - [A/B Testing Setups](#ab-testing-setups)
  - [Hyperparameter Sweeps](#hyperparameter-sweeps)

## Configuration Patterns

### Factory Functions

Factory functions allow you to create complex configurations dynamically based on input parameters. This pattern is especially useful when configurations have interdependencies or need runtime computation.

```python
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional
import torch
import torch.nn as nn
from dream_trainer import DreamTrainerConfig
from dream_trainer.configs import DeviceParameters, TrainingParameters

# Model factory function
def create_transformer_config(
    model_size: str = "base",
    vocab_size: int = 50257,
    max_seq_len: int = 2048,
    flash_attention: bool = True,
) -> Callable[[], nn.Module]:
    """Factory function for creating transformer models with size presets."""
    
    size_configs = {
        "tiny": {"dim": 384, "heads": 6, "layers": 6, "ffn_mult": 4},
        "small": {"dim": 768, "heads": 12, "layers": 12, "ffn_mult": 4},
        "base": {"dim": 1024, "heads": 16, "layers": 24, "ffn_mult": 4},
        "large": {"dim": 1280, "heads": 20, "layers": 36, "ffn_mult": 4},
        "xl": {"dim": 1600, "heads": 25, "layers": 48, "ffn_mult": 4},
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = size_configs[model_size]
    
    # Return a partial function that creates the model
    return partial(
        TransformerModel,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=config["dim"],
        n_heads=config["heads"],
        n_layers=config["layers"],
        ffn_dim_multiplier=config["ffn_mult"],
        use_flash_attention=flash_attention,
    )

# Optimizer factory with model-size-aware settings
def create_optimizer_config(
    model_size: str,
    base_lr: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
) -> Callable:
    """Create optimizer config with size-appropriate learning rates."""
    
    # Scale learning rate based on model size
    lr_scale = {
        "tiny": 2.0,
        "small": 1.5, 
        "base": 1.0,
        "large": 0.8,
        "xl": 0.6,
    }
    
    scaled_lr = base_lr * lr_scale.get(model_size, 1.0)
    
    return partial(
        torch.optim.AdamW,
        lr=scaled_lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        eps=1e-8,
    )

# Scheduler factory with warmup calculation
def create_scheduler_config(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a cosine scheduler with linear warmup."""
    
    warmup_steps = int(total_steps * warmup_ratio)
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress)) / 2
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Usage example
@dataclass
class ExperimentConfig(DreamTrainerConfig):
    """Config using factory functions for dynamic configuration."""
    
    # Model configuration
    model_size: str = "base"
    model: Callable = field(default_factory=lambda: create_transformer_config("base"))
    
    # Optimizer configuration 
    optimizer: Callable = field(default_factory=lambda: create_optimizer_config("base"))
    
    # Scheduler configuration (requires optimizer instance)
    scheduler_factory: Callable = field(
        default_factory=lambda: create_scheduler_config
    )
    
    def __post_init__(self):
        """Update factories based on model size."""
        self.model = create_transformer_config(self.model_size)
        self.optimizer = create_optimizer_config(self.model_size)
```

### Conditional Configuration

Conditional configuration allows you to adjust settings based on runtime conditions, hardware availability, or other factors.

```python
import os
from pathlib import Path
from typing import Optional, Union

@dataclass
class ConditionalConfig(DreamTrainerConfig):
    """Configuration that adapts to runtime environment."""
    
    # Basic settings
    experiment_name: str = "adaptive_training"
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "0") == "1")
    
    # Conditional batch size based on GPU memory
    @property
    def batch_size(self) -> int:
        """Dynamically set batch size based on available GPU memory."""
        if not torch.cuda.is_available():
            return 4  # CPU debugging
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if self.debug_mode:
            return 2
        elif gpu_memory_gb < 16:
            return 8
        elif gpu_memory_gb < 40:
            return 16
        elif gpu_memory_gb < 80:
            return 32
        else:
            return 64
    
    # Conditional gradient accumulation
    @property
    def gradient_accumulation_steps(self) -> int:
        """Adjust gradient accumulation to maintain effective batch size."""
        target_batch_size = 256  # Target effective batch size
        return max(1, target_batch_size // (self.batch_size * self.world_size))
    
    # Conditional precision settings
    @property
    def mixed_precision(self) -> Optional[str]:
        """Enable mixed precision based on GPU capability."""
        if not torch.cuda.is_available():
            return None
        
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):  # Ampere or newer
            return "bf16"
        elif capability >= (7, 0):  # Volta or newer
            return "fp16"
        else:
            return None
    
    # Conditional compilation
    @property
    def compile_model(self) -> bool:
        """Enable compilation only in production mode."""
        return not self.debug_mode and torch.cuda.is_available()
    
    # Conditional checkpointing
    @property
    def checkpoint_frequency(self) -> Union[int, float]:
        """Checkpoint more frequently in debug mode."""
        return 0.1 if self.debug_mode else 1.0  # Every 10% vs every epoch
    
    # Conditional data paths
    @property
    def data_root(self) -> Path:
        """Select data path based on environment."""
        if cluster_path := os.getenv("CLUSTER_DATA_PATH"):
            return Path(cluster_path)
        elif self.debug_mode:
            return Path("./debug_data")
        else:
            return Path("/data/datasets")

# Advanced conditional configuration with feature flags
@dataclass 
class FeatureFlagConfig(DreamTrainerConfig):
    """Configuration with feature flags for gradual rollout."""
    
    # Feature flags
    enable_new_optimizer: bool = field(
        default_factory=lambda: os.getenv("ENABLE_NEW_OPTIMIZER", "false").lower() == "true"
    )
    enable_experimental_parallelism: bool = field(
        default_factory=lambda: os.getenv("ENABLE_EXP_PARALLEL", "false").lower() == "true"
    )
    
    def configure_optimizer(self) -> Callable:
        """Select optimizer based on feature flag."""
        if self.enable_new_optimizer:
            # New experimental optimizer
            return partial(
                torch.optim.AdamW,
                lr=1e-4,
                betas=(0.95, 0.98),
                weight_decay=0.05,
                fused=True,  # New feature
            )
        else:
            # Stable optimizer
            return partial(
                torch.optim.AdamW,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
    
    def configure_device_parameters(self) -> DeviceParameters:
        """Configure parallelism based on feature flags."""
        if self.enable_experimental_parallelism:
            return DeviceParameters(
                dp_shard="auto",
                tensor_parallel=4,
                context_parallel=2,
                async_tensor_parallel=True,
                loss_parallel=True,
            )
        else:
            return DeviceParameters(
                dp_shard="auto",
                tensor_parallel=1,
                compile_model=True,
            )
```

### Environment-Based Configs

Environment-based configurations allow you to maintain different settings for development, staging, and production environments.

```python
import socket
from enum import Enum, auto
from typing import Dict, Any

class Environment(Enum):
    """Training environment types."""
    LOCAL = auto()
    DEVELOPMENT = auto() 
    STAGING = auto()
    PRODUCTION = auto()

def detect_environment() -> Environment:
    """Automatically detect the current environment."""
    hostname = socket.gethostname()
    
    if "local" in hostname or "laptop" in hostname:
        return Environment.LOCAL
    elif "dev-" in hostname:
        return Environment.DEVELOPMENT
    elif "staging-" in hostname:
        return Environment.STAGING
    elif "prod-" in hostname or "cluster" in hostname:
        return Environment.PRODUCTION
    else:
        # Check environment variable
        env_name = os.getenv("TRAINING_ENV", "local").lower()
        return Environment[env_name.upper()]

@dataclass
class EnvironmentConfig(DreamTrainerConfig):
    """Configuration that adapts to different environments."""
    
    environment: Environment = field(default_factory=detect_environment)
    
    # Environment-specific settings
    ENV_SETTINGS: Dict[Environment, Dict[str, Any]] = field(default_factory=lambda: {
        Environment.LOCAL: {
            "num_workers": 0,  # Avoid multiprocessing issues
            "prefetch_factor": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "log_frequency": 1,  # Log every step
            "checkpoint_frequency": 0.1,  # Frequent checkpoints
            "enable_profiling": True,
            "compile_model": False,
            "use_wandb": False,
        },
        Environment.DEVELOPMENT: {
            "num_workers": 4,
            "prefetch_factor": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "log_frequency": 10,
            "checkpoint_frequency": 0.25,
            "enable_profiling": True,
            "compile_model": False,
            "use_wandb": True,
        },
        Environment.STAGING: {
            "num_workers": 8,
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "log_frequency": 50,
            "checkpoint_frequency": 1.0,
            "enable_profiling": False,
            "compile_model": True,
            "use_wandb": True,
        },
        Environment.PRODUCTION: {
            "num_workers": 16,
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "log_frequency": 100,
            "checkpoint_frequency": 1.0,
            "enable_profiling": False,
            "compile_model": True,
            "use_wandb": True,
        },
    })
    
    def __post_init__(self):
        """Apply environment-specific settings."""
        settings = self.ENV_SETTINGS[self.environment]
        
        # Apply settings dynamically
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @property
    def data_path(self) -> Path:
        """Environment-specific data paths."""
        paths = {
            Environment.LOCAL: Path("./data/sample"),
            Environment.DEVELOPMENT: Path("/mnt/dev-data"),
            Environment.STAGING: Path("/mnt/staging-data"),
            Environment.PRODUCTION: Path("/mnt/production-data"),
        }
        return paths[self.environment]
    
    @property
    def checkpoint_dir(self) -> Path:
        """Environment-specific checkpoint directories."""
        base_paths = {
            Environment.LOCAL: Path("./checkpoints"),
            Environment.DEVELOPMENT: Path("/mnt/dev-checkpoints"),
            Environment.STAGING: Path("/mnt/staging-checkpoints"),
            Environment.PRODUCTION: Path("/mnt/production-checkpoints"),
        }
        return base_paths[self.environment] / self.experiment

# Cluster-aware configuration
@dataclass
class ClusterConfig(EnvironmentConfig):
    """Configuration aware of specific cluster environments."""
    
    @property
    def cluster_type(self) -> str:
        """Detect cluster type from environment."""
        if slurm_cluster := os.getenv("SLURM_CLUSTER_NAME"):
            return f"slurm_{slurm_cluster}"
        elif k8s_namespace := os.getenv("K8S_NAMESPACE"):
            return f"k8s_{k8s_namespace}"
        elif aws_region := os.getenv("AWS_DEFAULT_REGION"):
            return f"aws_{aws_region}"
        else:
            return "local"
    
    def configure_for_cluster(self) -> DeviceParameters:
        """Configure device parameters based on cluster."""
        if "slurm" in self.cluster_type:
            # SLURM cluster configuration
            return DeviceParameters(
                dp_shard="auto",
                comm={"backend": "nccl", "timeout": 1800},
                cpu_offload=False,
            )
        elif "k8s" in self.cluster_type:
            # Kubernetes cluster configuration
            return DeviceParameters(
                dp_shard="auto",
                comm={"backend": "nccl", "init_method": "env://"},
                checkpoint_activations=True,  # Memory constrained
            )
        elif "aws" in self.cluster_type:
            # AWS cluster configuration
            return DeviceParameters(
                dp_shard="auto",
                comm={"backend": "nccl", "timeout": 3600},
                enable_compiled_autograd=True,
            )
        else:
            # Local configuration
            return DeviceParameters(
                dp_shard=1,
                compile_model=False,
            )
```

### Multi-Experiment Management

Managing multiple experiments with shared base configurations and experiment-specific overrides.

```python
from typing import List, Dict, Type
from datetime import datetime

@dataclass
class BaseExperimentConfig(DreamTrainerConfig):
    """Base configuration shared across all experiments."""
    
    # Shared project settings
    project: str = "transformer_research"
    entity: str = "my_team"
    tags: List[str] = field(default_factory=lambda: ["baseline", "v2"])
    
    # Shared model architecture
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Shared training settings
    gradient_clip_val: float = 1.0
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    
    # Shared data settings
    data_root: Path = Path("/data/common_dataset")
    
    def experiment_name(self) -> str:
        """Generate unique experiment name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.group}_{timestamp}"

# Experiment registry
EXPERIMENTS: Dict[str, Type[BaseExperimentConfig]] = {}

def register_experiment(name: str):
    """Decorator to register experiments."""
    def decorator(cls):
        EXPERIMENTS[name] = cls
        return cls
    return decorator

@register_experiment("small_baseline")
@dataclass
class SmallBaselineConfig(BaseExperimentConfig):
    """Small model for quick iteration."""
    
    group: str = "small_baseline"
    model_size: str = "small"
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    
    # Small model specific settings
    learning_rate: float = 6e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Frequent validation for debugging
    val_frequency: float = 0.1
    
    # More tags
    tags: List[str] = field(
        default_factory=lambda: BaseExperimentConfig.tags.__wrapped__() + ["small", "debug"]
    )

@register_experiment("large_scale")
@dataclass
class LargeScaleConfig(BaseExperimentConfig):
    """Large model for production training."""
    
    group: str = "large_scale"
    model_size: str = "large"
    hidden_size: int = 1280
    num_heads: int = 20
    num_layers: int = 36
    
    # Large model specific settings
    learning_rate: float = 2e-4
    batch_size: int = 8  # Per GPU
    gradient_accumulation_steps: int = 32  # Effective batch size 256
    
    # Less frequent validation to save time
    val_frequency: float = 1.0
    
    # Enable optimizations
    compile_model: bool = True
    checkpoint_activations: bool = True
    
    # Parallelism settings
    device_parameters: DeviceParameters = field(
        default_factory=lambda: DeviceParameters(
            dp_shard="auto",
            tensor_parallel=4,
            loss_parallel=True,
        )
    )

@register_experiment("ablation_no_warmup")
@dataclass 
class AblationNoWarmupConfig(BaseExperimentConfig):
    """Ablation study: no learning rate warmup."""
    
    group: str = "ablation_studies"
    warmup_steps: int = 0  # Override base config
    
    # Keep most settings from base
    model_size: str = "base"
    hidden_size: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    
    tags: List[str] = field(
        default_factory=lambda: BaseExperimentConfig.tags.__wrapped__() + ["ablation", "no_warmup"]
    )

@register_experiment("hyperparameter_search")
@dataclass
class HyperparameterSearchConfig(BaseExperimentConfig):
    """Configuration for hyperparameter search."""
    
    group: str = "hyperparam_search"
    
    # Hyperparameters to search
    learning_rate: float = field(default_factory=lambda: np.random.choice([1e-4, 3e-4, 6e-4]))
    dropout: float = field(default_factory=lambda: np.random.uniform(0.0, 0.3))
    weight_decay: float = field(default_factory=lambda: np.random.choice([0.01, 0.05, 0.1]))
    
    # Fixed model size for fair comparison
    model_size: str = "base"
    hidden_size: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    
    def __post_init__(self):
        """Add hyperparameter values to tags."""
        super().__post_init__()
        self.tags.extend([
            f"lr_{self.learning_rate}",
            f"dropout_{self.dropout:.2f}",
            f"wd_{self.weight_decay}",
        ])

# Experiment launcher
def launch_experiment(experiment_name: str, **overrides):
    """Launch an experiment by name with optional overrides."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    config_class = EXPERIMENTS[experiment_name]
    config = config_class(**overrides)
    
    # Create trainer and run
    trainer = DreamTrainer(config)
    trainer.fit()
    
    return trainer

# Multi-experiment runner
def run_all_experiments():
    """Run all registered experiments."""
    results = {}
    
    for name, config_class in EXPERIMENTS.items():
        print(f"Running experiment: {name}")
        config = config_class()
        
        trainer = DreamTrainer(config)
        trainer.fit()
        
        results[name] = {
            "config": config,
            "metrics": trainer.get_metrics(),
        }
    
    return results
```

## Validation & Constraints

### Custom Validators

Implement custom validation logic to ensure configuration consistency and catch errors early.

```python
from typing import Union, List, Optional
import re

class ValidationError(ValueError):
    """Custom exception for configuration validation errors."""
    pass

@dataclass
class ValidatedConfig(DreamTrainerConfig):
    """Configuration with comprehensive validation."""
    
    # Model settings with validation
    model_name: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    # Training settings
    learning_rate: float = 3e-4
    batch_size: int = 32
    gradient_clip_val: Optional[float] = 1.0
    
    # Paths
    checkpoint_path: Optional[Path] = None
    
    def __post_init__(self):
        """Run all validations after initialization."""
        self.validate_model_architecture()
        self.validate_training_parameters()
        self.validate_paths()
        self.validate_dependencies()
    
    def validate_model_architecture(self):
        """Validate model architecture consistency."""
        # Check if hidden size is divisible by num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValidationError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        # Check reasonable layer count
        if not 1 <= self.num_layers <= 200:
            raise ValidationError(
                f"num_layers ({self.num_layers}) should be between 1 and 200"
            )
        
        # Validate model name format
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", self.model_name):
            raise ValidationError(
                f"Invalid model name: {self.model_name}. "
                "Must start with letter and contain only alphanumeric, dash, or underscore"
            )
    
    def validate_training_parameters(self):
        """Validate training hyperparameters."""
        # Learning rate range
        if not 1e-6 <= self.learning_rate <= 1e-1:
            raise ValidationError(
                f"learning_rate ({self.learning_rate}) should be between 1e-6 and 1e-1"
            )
        
        # Batch size must be positive
        if self.batch_size < 1:
            raise ValidationError(f"batch_size must be positive, got {self.batch_size}")
        
        # Gradient clipping validation
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValidationError(
                f"gradient_clip_val must be positive or None, got {self.gradient_clip_val}"
            )
    
    def validate_paths(self):
        """Validate file paths and permissions."""
        if self.checkpoint_path is not None:
            # Ensure parent directory exists
            parent = self.checkpoint_path.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(
                        f"Cannot create checkpoint directory {parent}: {e}"
                    )
            
            # Check write permissions
            if parent.exists() and not os.access(parent, os.W_OK):
                raise ValidationError(
                    f"No write permission for checkpoint directory: {parent}"
                )
    
    def validate_dependencies(self):
        """Validate dependencies between configuration options."""
        # If using mixed precision, ensure compatible device
        if hasattr(self, "mixed_precision") and self.mixed_precision:
            if not torch.cuda.is_available():
                raise ValidationError(
                    "Mixed precision training requires CUDA device"
                )
            
            capability = torch.cuda.get_device_capability()
            if self.mixed_precision == "fp16" and capability < (7, 0):
                raise ValidationError(
                    f"FP16 requires GPU capability >= 7.0, got {capability}"
                )

# Advanced constraint validation
@dataclass
class ConstrainedConfig(DreamTrainerConfig):
    """Configuration with complex constraints."""
    
    # Constrained values
    learning_rate_schedule: str = "cosine"
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-3
    
    # Dependent parameters
    use_weight_decay: bool = True
    weight_decay: float = 0.1
    
    # Resource constraints
    max_memory_gb: float = 40.0
    model_size: str = "large"
    batch_size: int = 32
    
    VALID_SCHEDULES = ["cosine", "linear", "constant", "exponential"]
    
    MODEL_MEMORY_REQUIREMENTS = {
        "small": 2.0,  # GB
        "base": 6.0,
        "large": 18.0,
        "xl": 40.0,
    }
    
    def validate_constraints(self):
        """Validate complex constraints between parameters."""
        # Schedule validation
        if self.learning_rate_schedule not in self.VALID_SCHEDULES:
            raise ValidationError(
                f"Invalid schedule: {self.learning_rate_schedule}. "
                f"Must be one of {self.VALID_SCHEDULES}"
            )
        
        # Learning rate ordering
        if self.min_learning_rate >= self.max_learning_rate:
            raise ValidationError(
                f"min_learning_rate ({self.min_learning_rate}) must be less than "
                f"max_learning_rate ({self.max_learning_rate})"
            )
        
        # Weight decay dependency
        if not self.use_weight_decay and self.weight_decay != 0:
            raise ValidationError(
                "weight_decay must be 0 when use_weight_decay is False"
            )
        
        # Memory constraints
        model_memory = self.MODEL_MEMORY_REQUIREMENTS.get(self.model_size, 0)
        batch_memory = model_memory * self.batch_size / 32  # Scale by batch size
        
        if batch_memory > self.max_memory_gb:
            raise ValidationError(
                f"Configuration requires ~{batch_memory:.1f}GB memory, "
                f"but max_memory_gb is {self.max_memory_gb}. "
                f"Reduce batch_size or use smaller model."
            )
```

### Type Safety Patterns

Leverage Python's type system for compile-time safety and better IDE support.

```python
from typing import TypeVar, Generic, Protocol, runtime_checkable
from typing import Literal, Union, get_type_hints, get_origin, get_args

T = TypeVar('T')

@runtime_checkable
class Configurable(Protocol):
    """Protocol for objects that can be configured."""
    
    def configure(self, config: Any) -> None:
        """Apply configuration to the object."""
        ...

@dataclass
class TypedConfig(Generic[T]):
    """Generic typed configuration container."""
    
    value: T
    
    def __post_init__(self):
        """Validate type at runtime."""
        expected_type = get_type_hints(self.__class__)['value']
        
        if hasattr(expected_type, '__origin__'):
            # Handle generic types like List[int], Dict[str, float], etc.
            origin = get_origin(expected_type)
            args = get_args(expected_type)
            
            if not isinstance(self.value, origin):
                raise TypeError(
                    f"Expected {origin}, got {type(self.value)}"
                )
            
            # Additional validation for container types
            if origin is list and args:
                item_type = args[0]
                for item in self.value:
                    if not isinstance(item, item_type):
                        raise TypeError(
                            f"List item {item} is not of type {item_type}"
                        )
        else:
            # Simple type check
            if not isinstance(self.value, expected_type):
                raise TypeError(
                    f"Expected {expected_type}, got {type(self.value)}"
                )

# Strongly typed model configuration
@dataclass
class ModelConfig:
    """Strongly typed model configuration."""
    
    architecture: Literal["transformer", "cnn", "rnn"]
    hidden_size: int
    num_layers: int
    activation: Literal["relu", "gelu", "swish"]
    
    # Use NewType for additional type safety
    from typing import NewType
    
    Dimension = NewType('Dimension', int)
    BatchSize = NewType('BatchSize', int)
    
    def validate_positive(self, value: int, name: str) -> None:
        """Ensure positive integer values."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    def __post_init__(self):
        """Validate all integer parameters."""
        self.validate_positive(self.hidden_size, "hidden_size")
        self.validate_positive(self.num_layers, "num_layers")

# Type-safe configuration builder
class ConfigBuilder(Generic[T]):
    """Fluent interface for building configurations."""
    
    def __init__(self, config_class: Type[T]):
        self.config_class = config_class
        self.params = {}
    
    def set(self, **kwargs) -> 'ConfigBuilder[T]':
        """Set configuration parameters."""
        # Validate against class annotations
        hints = get_type_hints(self.config_class)
        
        for key, value in kwargs.items():
            if key not in hints:
                raise ValueError(f"Unknown parameter: {key}")
            
            expected_type = hints[key]
            # Basic type checking (can be extended)
            if not self._check_type(value, expected_type):
                raise TypeError(
                    f"Parameter {key} expects {expected_type}, got {type(value)}"
                )
            
            self.params[key] = value
        
        return self
    
    def _check_type(self, value: Any, expected_type: Type) -> bool:
        """Check if value matches expected type."""
        # Handle Optional types
        if get_origin(expected_type) is Union:
            args = get_args(expected_type)
            return any(self._check_type(value, arg) for arg in args)
        
        # Handle Literal types
        if get_origin(expected_type) is Literal:
            return value in get_args(expected_type)
        
        # Simple instance check
        return isinstance(value, expected_type)
    
    def build(self) -> T:
        """Build the configuration object."""
        return self.config_class(**self.params)

# Usage example
config = (ConfigBuilder(ModelConfig)
    .set(architecture="transformer")
    .set(hidden_size=768)
    .set(num_layers=12)
    .set(activation="gelu")
    .build())
```

### Configuration Testing

Implement comprehensive testing for your configurations to ensure they work correctly across different scenarios.

```python
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json

class ConfigTestCase(unittest.TestCase):
    """Base class for configuration testing."""
    
    def assertConfigValid(self, config: Any) -> None:
        """Assert that a configuration is valid."""
        try:
            # Attempt to create trainer with config
            trainer = DreamTrainer(config)
            # Attempt to access all properties
            _ = config.__dict__
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
    
    def assertConfigInvalid(self, config_fn: Callable, error_pattern: str) -> None:
        """Assert that a configuration raises expected error."""
        with self.assertRaisesRegex(ValidationError, error_pattern):
            config_fn()

class TestModelConfiguration(ConfigTestCase):
    """Test model configuration validation."""
    
    def test_valid_configuration(self):
        """Test that valid configurations pass."""
        config = ValidatedConfig(
            model_name="test_model",
            hidden_size=768,
            num_heads=12,
            num_layers=12,
        )
        self.assertConfigValid(config)
    
    def test_invalid_hidden_size(self):
        """Test that invalid hidden size is caught."""
        self.assertConfigInvalid(
            lambda: ValidatedConfig(
                hidden_size=769,  # Not divisible by 12
                num_heads=12,
            ),
            "must be divisible by"
        )
    
    def test_invalid_layer_count(self):
        """Test that invalid layer count is caught."""
        self.assertConfigInvalid(
            lambda: ValidatedConfig(
                num_layers=300,  # Too many
            ),
            "should be between"
        )

class TestEnvironmentConfiguration(ConfigTestCase):
    """Test environment-based configuration."""
    
    @patch.dict(os.environ, {"TRAINING_ENV": "production"})
    def test_production_environment(self):
        """Test production environment settings."""
        config = EnvironmentConfig()
        
        self.assertEqual(config.environment, Environment.PRODUCTION)
        self.assertTrue(config.compile_model)
        self.assertEqual(config.num_workers, 16)
        self.assertTrue(config.use_wandb)
    
    @patch.dict(os.environ, {"TRAINING_ENV": "local"})
    def test_local_environment(self):
        """Test local environment settings."""
        config = EnvironmentConfig()
        
        self.assertEqual(config.environment, Environment.LOCAL)
        self.assertFalse(config.compile_model)
        self.assertEqual(config.num_workers, 0)
        self.assertFalse(config.use_wandb)
    
    @patch.dict(os.environ, {"DEBUG": "1"})
    def test_debug_mode(self):
        """Test debug mode configuration."""
        config = ConditionalConfig()
        
        self.assertTrue(config.debug_mode)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.checkpoint_frequency, 0.1)

class TestConfigurationSerialization(ConfigTestCase):
    """Test configuration serialization/deserialization."""
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = BaseExperimentConfig(
            project="test_project",
            group="test_group",
            learning_rate=1e-4,
        )
        
        config_dict = asdict(config)
        
        self.assertEqual(config_dict["project"], "test_project")
        self.assertEqual(config_dict["learning_rate"], 1e-4)
    
    def test_config_from_json(self):
        """Test loading configuration from JSON."""
        config_data = {
            "project": "test_project",
            "group": "test_group",
            "experiment": "test_exp",
            "learning_rate": 0.0001,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(config_data, f)
            f.flush()
            
            # Load config from file
            loaded_data = json.load(open(f.name))
            config = BaseExperimentConfig(**loaded_data)
            
            self.assertEqual(config.project, "test_project")
            self.assertEqual(config.learning_rate, 0.0001)

class TestConfigurationIntegration(ConfigTestCase):
    """Integration tests for configurations."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_gpu_adaptive_config(self, mock_props, mock_available):
        """Test GPU-adaptive configuration."""
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.total_memory = 40 * 1e9  # 40GB
        mock_props.return_value = mock_device
        
        config = ConditionalConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertIsNotNone(config.mixed_precision)
    
    def test_multi_experiment_configs(self):
        """Test multiple experiment configurations."""
        configs = [
            SmallBaselineConfig(),
            LargeScaleConfig(),
            AblationNoWarmupConfig(),
        ]
        
        # Ensure all configs are valid
        for config in configs:
            self.assertConfigValid(config)
        
        # Ensure they have different settings
        self.assertNotEqual(configs[0].learning_rate, configs[1].learning_rate)
        self.assertNotEqual(configs[0].num_layers, configs[1].num_layers)
        self.assertEqual(configs[2].warmup_steps, 0)

# Property-based testing with hypothesis
from hypothesis import given, strategies as st, assume

class TestPropertyBasedConfig(ConfigTestCase):
    """Property-based testing for configurations."""
    
    @given(
        hidden_size=st.integers(min_value=64, max_value=4096),
        num_heads=st.integers(min_value=1, max_value=64),
    )
    def test_hidden_size_divisibility(self, hidden_size, num_heads):
        """Test that valid hidden sizes are always divisible by num_heads."""
        assume(hidden_size % num_heads == 0)
        
        config = ValidatedConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        
        self.assertConfigValid(config)
    
    @given(
        lr=st.floats(min_value=1e-6, max_value=1e-1),
        batch_size=st.integers(min_value=1, max_value=512),
    )
    def test_training_parameters_ranges(self, lr, batch_size):
        """Test that training parameters within valid ranges work."""
        config = ValidatedConfig(
            learning_rate=lr,
            batch_size=batch_size,
        )
        
        self.assertConfigValid(config)
```

## Real-World Examples

### Multi-Node Configurations

Configure training across multiple nodes with proper fault tolerance and communication settings.

```python
@dataclass
class MultiNodeConfig(DreamTrainerConfig):
    """Configuration for multi-node distributed training."""
    
    # Node configuration
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # Communication settings
    nccl_debug: str = field(default_factory=lambda: os.getenv("NCCL_DEBUG", "INFO"))
    nccl_timeout: int = 1800  # 30 minutes
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    checkpoint_frequency: float = 0.25  # Every 25% of epoch
    resume_from_checkpoint: bool = True
    
    def __post_init__(self):
        """Configure multi-node settings."""
        # Set environment variables for PyTorch distributed
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.num_nodes * self.gpus_per_node)
        os.environ["RANK"] = str(self.node_rank * self.gpus_per_node + self.local_rank)
        
        # NCCL optimizations
        os.environ["NCCL_DEBUG"] = self.nccl_debug
        os.environ["NCCL_TREE_THRESHOLD"] = "0"
        
        # InfiniBand optimizations (if available)
        if self.has_infiniband():
            os.environ["NCCL_IB_DISABLE"] = "0"
            os.environ["NCCL_IB_GID_INDEX"] = "3"
            os.environ["NCCL_IB_TIMEOUT"] = "22"
        
        # Configure device parameters for multi-node
        self.device_parameters = self.configure_multi_node_devices()
    
    @property
    def gpus_per_node(self) -> int:
        """Get number of GPUs per node."""
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    @property
    def local_rank(self) -> int:
        """Get local rank within the node."""
        return int(os.getenv("LOCAL_RANK", 0))
    
    def has_infiniband(self) -> bool:
        """Check if InfiniBand is available."""
        try:
            import subprocess
            result = subprocess.run(["ibstat"], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def configure_multi_node_devices(self) -> DeviceParameters:
        """Configure device parameters for multi-node training."""
        total_gpus = self.num_nodes * self.gpus_per_node
        
        # Automatic sharding configuration
        if total_gpus >= 64:
            # Large scale: use HSDP
            return DeviceParameters(
                dp_shard=8,  # FSDP within groups of 8
                dp_replicate=total_gpus // 8,  # Replicate across groups
                tensor_parallel=1,
                compile_model=True,
                comm={
                    "backend": "nccl",
                    "timeout": self.nccl_timeout,
                },
            )
        elif total_gpus >= 16:
            # Medium scale: use FSDP with TP
            return DeviceParameters(
                dp_shard="auto",
                tensor_parallel=4,
                compile_model=True,
                async_tensor_parallel=True,
                comm={
                    "backend": "nccl",
                    "timeout": self.nccl_timeout,
                },
            )
        else:
            # Small scale: simple FSDP
            return DeviceParameters(
                dp_shard="auto",
                compile_model=True,
                comm={
                    "backend": "nccl",
                    "timeout": self.nccl_timeout,
                },
            )

# SLURM-specific configuration
@dataclass
class SlurmConfig(MultiNodeConfig):
    """Configuration for SLURM-managed clusters."""
    
    def __post_init__(self):
        """Auto-detect SLURM environment."""
        if slurm_nodeid := os.getenv("SLURM_NODEID"):
            self.node_rank = int(slurm_nodeid)
        
        if slurm_nnodes := os.getenv("SLURM_NNODES"):
            self.num_nodes = int(slurm_nnodes)
        
        if slurm_step_nodelist := os.getenv("SLURM_STEP_NODELIST"):
            # Parse first node as master
            import re
            match = re.match(r'([^,\[]+)', slurm_step_nodelist)
            if match:
                self.master_addr = match.group(1)
        
        # Set port based on job ID to avoid conflicts
        if slurm_jobid := os.getenv("SLURM_JOBID"):
            self.master_port = 29500 + int(slurm_jobid) % 1000
        
        super().__post_init__()
```

### A/B Testing Setups

Configure experiments for A/B testing different model architectures, training strategies, or hyperparameters.

```python
@dataclass
class ABTestConfig(DreamTrainerConfig):
    """Base configuration for A/B testing."""
    
    # Test identification
    test_name: str
    variant: Literal["A", "B"]
    
    # Shared settings for fair comparison
    random_seed: int = 42
    dataset_seed: int = 123
    num_epochs: int = 10
    eval_metric: str = "perplexity"
    
    # Logging for comparison
    use_wandb: bool = True
    wandb_project: str = "ab_tests"
    
    def __post_init__(self):
        """Set up variant-specific configuration."""
        self.experiment = f"{self.test_name}_{self.variant}"
        self.wandb_group = self.test_name
        self.wandb_tags = [self.test_name, f"variant_{self.variant}"]

@dataclass
class ArchitectureABTest(ABTestConfig):
    """A/B test comparing different architectures."""
    
    test_name: str = "architecture_comparison"
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get variant-specific model configuration."""
        if self.variant == "A":
            # Variant A: Traditional transformer
            return {
                "architecture": "transformer",
                "hidden_size": 768,
                "num_heads": 12,
                "num_layers": 12,
                "ffn_multiplier": 4,
                "activation": "gelu",
                "use_rope": False,
            }
        else:
            # Variant B: Modern improvements
            return {
                "architecture": "transformer",
                "hidden_size": 768,
                "num_heads": 12,
                "num_layers": 12,
                "ffn_multiplier": 3,  # SwiGLU uses 3x
                "activation": "swiglu",
                "use_rope": True,
                "rope_theta": 10000,
            }

@dataclass
class OptimizerABTest(ABTestConfig):
    """A/B test comparing different optimizers."""
    
    test_name: str = "optimizer_comparison"
    
    # Same model for both variants
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 12,
    })
    
    def get_optimizer_config(self) -> Callable:
        """Get variant-specific optimizer configuration."""
        if self.variant == "A":
            # Variant A: Standard AdamW
            return partial(
                torch.optim.AdamW,
                lr=3e-4,
                betas=(0.9, 0.999),
                weight_decay=0.1,
                eps=1e-8,
            )
        else:
            # Variant B: Modified AdamW with different betas
            return partial(
                torch.optim.AdamW,
                lr=3e-4,
                betas=(0.95, 0.98),  # GPT-3 style
                weight_decay=0.05,
                eps=1e-8,
            )

@dataclass
class TrainingStrategyABTest(ABTestConfig):
    """A/B test comparing different training strategies."""
    
    test_name: str = "training_strategy"
    
    def configure_training(self) -> Dict[str, Any]:
        """Get variant-specific training configuration."""
        if self.variant == "A":
            # Variant A: Standard training
            return {
                "gradient_accumulation_steps": 1,
                "gradient_clip_val": 1.0,
                "warmup_steps": 2000,
                "scheduler": "cosine",
                "use_mixed_precision": True,
                "mixed_precision_type": "fp16",
            }
        else:
            # Variant B: Advanced training
            return {
                "gradient_accumulation_steps": 4,
                "gradient_clip_val": 0.5,
                "warmup_steps": 4000,
                "scheduler": "cosine_with_restarts",
                "use_mixed_precision": True,
                "mixed_precision_type": "bf16",
                "use_gradient_checkpointing": True,
            }

# A/B test runner
class ABTestRunner:
    """Runner for A/B tests with statistical analysis."""
    
    def __init__(self, test_config_class: Type[ABTestConfig]):
        self.test_config_class = test_config_class
        self.results = {"A": [], "B": []}
    
    def run_test(self, num_runs: int = 3) -> Dict[str, Any]:
        """Run A/B test with multiple seeds."""
        for run in range(num_runs):
            for variant in ["A", "B"]:
                config = self.test_config_class(
                    variant=variant,
                    random_seed=42 + run,  # Different seed per run
                )
                
                # Run training
                trainer = DreamTrainer(config)
                trainer.fit()
                
                # Collect metrics
                final_metric = trainer.get_metric(config.eval_metric)
                self.results[variant].append(final_metric)
        
        # Statistical analysis
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Perform statistical analysis of A/B test results."""
        from scipy import stats
        import numpy as np
        
        a_results = np.array(self.results["A"])
        b_results = np.array(self.results["B"])
        
        # T-test
        t_stat, p_value = stats.ttest_ind(a_results, b_results)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(a_results) + np.var(b_results)) / 2)
        effect_size = (np.mean(b_results) - np.mean(a_results)) / pooled_std
        
        return {
            "variant_A": {
                "mean": np.mean(a_results),
                "std": np.std(a_results),
                "results": a_results.tolist(),
            },
            "variant_B": {
                "mean": np.mean(b_results),
                "std": np.std(b_results),
                "results": b_results.tolist(),
            },
            "statistics": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "significant": p_value < 0.05,
                "better_variant": "B" if np.mean(b_results) < np.mean(a_results) else "A",
            },
        }
```

### Hyperparameter Sweeps

Configure and run hyperparameter sweeps with different search strategies.

```python
from typing import Tuple
import itertools
import random

@dataclass
class SweepConfig(DreamTrainerConfig):
    """Base configuration for hyperparameter sweeps."""
    
    sweep_name: str
    sweep_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Hyperparameter bounds
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-3)
    batch_size_options: List[int] = field(default_factory=lambda: [16, 32, 64])
    weight_decay_range: Tuple[float, float] = (0.0, 0.3)
    warmup_ratio_range: Tuple[float, float] = (0.0, 0.1)
    
    # Search configuration
    search_strategy: Literal["grid", "random", "bayesian"] = "random"
    num_trials: int = 20
    
    # Tracking
    use_wandb: bool = True
    wandb_project: str = "hyperparameter_sweeps"

class GridSearchConfig(SweepConfig):
    """Grid search configuration."""
    
    search_strategy: Literal["grid", "random", "bayesian"] = "grid"
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        # Define discrete values for grid
        learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        batch_sizes = self.batch_size_options
        weight_decays = [0.0, 0.01, 0.1, 0.2]
        warmup_ratios = [0.0, 0.05, 0.1]
        
        # Generate all combinations
        combinations = itertools.product(
            learning_rates,
            batch_sizes,
            weight_decays,
            warmup_ratios,
        )
        
        configs = []
        for lr, bs, wd, wr in combinations:
            configs.append({
                "learning_rate": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "warmup_ratio": wr,
                "config_id": f"lr{lr}_bs{bs}_wd{wd}_wr{wr}",
            })
        
        return configs

class RandomSearchConfig(SweepConfig):
    """Random search configuration."""
    
    search_strategy: Literal["grid", "random", "bayesian"] = "random"
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate random configurations."""
        configs = []
        
        for i in range(self.num_trials):
            # Sample hyperparameters
            lr = 10 ** random.uniform(
                np.log10(self.learning_rate_range[0]),
                np.log10(self.learning_rate_range[1])
            )
            bs = random.choice(self.batch_size_options)
            wd = random.uniform(*self.weight_decay_range)
            wr = random.uniform(*self.warmup_ratio_range)
            
            configs.append({
                "learning_rate": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "warmup_ratio": wr,
                "config_id": f"trial_{i:03d}",
            })
        
        return configs

class BayesianSearchConfig(SweepConfig):
    """Bayesian optimization configuration."""
    
    search_strategy: Literal["grid", "random", "bayesian"] = "bayesian"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize Bayesian optimization
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        
        self.search_space = [
            Real(np.log10(self.learning_rate_range[0]), 
                 np.log10(self.learning_rate_range[1]), 
                 name='log_learning_rate'),
            Categorical(self.batch_size_options, name='batch_size'),
            Real(*self.weight_decay_range, name='weight_decay'),
            Real(*self.warmup_ratio_range, name='warmup_ratio'),
        ]
        
        self.optimization_results = []
    
    def suggest_config(self) -> Dict[str, Any]:
        """Suggest next configuration using Bayesian optimization."""
        if len(self.optimization_results) < 5:
            # Random exploration for first few trials
            return RandomSearchConfig.generate_configs(self)[0]
        
        # Use Gaussian Process to suggest next point
        from skopt import gp_minimize
        
        X = [r['params'] for r in self.optimization_results]
        y = [r['objective'] for r in self.optimization_results]
        
        # Get suggestion (this is simplified, real implementation would be more complex)
        suggestion = self._get_suggestion(X, y)
        
        return {
            "learning_rate": 10 ** suggestion[0],
            "batch_size": suggestion[1],
            "weight_decay": suggestion[2],
            "warmup_ratio": suggestion[3],
            "config_id": f"bayes_{len(self.optimization_results):03d}",
        }
    
    def update_results(self, config: Dict[str, Any], objective: float):
        """Update Bayesian optimization with results."""
        self.optimization_results.append({
            "params": [
                np.log10(config["learning_rate"]),
                config["batch_size"],
                config["weight_decay"],
                config["warmup_ratio"],
            ],
            "objective": objective,
        })

# Hyperparameter sweep runner
class SweepRunner:
    """Runner for hyperparameter sweeps."""
    
    def __init__(self, base_config: Type[DreamTrainerConfig], sweep_config: SweepConfig):
        self.base_config = base_config
        self.sweep_config = sweep_config
        self.results = []
    
    def run_sweep(self) -> pd.DataFrame:
        """Run the complete hyperparameter sweep."""
        if self.sweep_config.search_strategy == "grid":
            configs = GridSearchConfig(**asdict(self.sweep_config)).generate_configs()
        elif self.sweep_config.search_strategy == "random":
            configs = RandomSearchConfig(**asdict(self.sweep_config)).generate_configs()
        else:
            # Bayesian optimization
            bayes_config = BayesianSearchConfig(**asdict(self.sweep_config))
            configs = []
            
            for _ in range(self.sweep_config.num_trials):
                config = bayes_config.suggest_config()
                result = self.run_trial(config)
                bayes_config.update_results(config, result["objective"])
                configs.append(config)
        
        # Run all trials
        for config in configs:
            result = self.run_trial(config)
            self.results.append(result)
        
        # Convert to DataFrame for analysis
        return pd.DataFrame(self.results)
    
    def run_trial(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single trial with given hyperparameters."""
        # Create configuration
        config = self.base_config(
            experiment=f"{self.sweep_config.sweep_name}_{hyperparams['config_id']}",
            learning_rate=hyperparams["learning_rate"],
            batch_size=hyperparams["batch_size"], 
            weight_decay=hyperparams["weight_decay"],
            warmup_ratio=hyperparams["warmup_ratio"],
            wandb_group=self.sweep_config.sweep_name,
            wandb_tags=[self.sweep_config.sweep_name, hyperparams['config_id']],
        )
        
        # Train model
        trainer = DreamTrainer(config)
        trainer.fit()
        
        # Get final metrics
        final_loss = trainer.get_metric("val_loss")
        final_perplexity = trainer.get_metric("perplexity")
        
        return {
            **hyperparams,
            "final_loss": final_loss,
            "final_perplexity": final_perplexity,
            "objective": final_perplexity,  # Metric to optimize
            "training_time": trainer.training_time,
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze sweep results."""
        df = pd.DataFrame(self.results)
        
        # Find best configuration
        best_idx = df["objective"].argmin()
        best_config = df.iloc[best_idx]
        
        # Analyze parameter importance
        correlations = {}
        for param in ["learning_rate", "batch_size", "weight_decay", "warmup_ratio"]:
            if param in df.columns:
                correlations[param] = df[param].corr(df["objective"])
        
        return {
            "best_config": best_config.to_dict(),
            "parameter_correlations": correlations,
            "results_summary": {
                "mean_objective": df["objective"].mean(),
                "std_objective": df["objective"].std(),
                "best_objective": df["objective"].min(),
                "worst_objective": df["objective"].max(),
            },
        }
```

## Summary

This configuration mastery guide demonstrates how Dream Trainer's Python-based configuration system enables:

1. **Dynamic Configuration**: Use factory functions and runtime computation to create adaptive configurations
2. **Type Safety**: Leverage Python's type system for compile-time checks and better IDE support
3. **Validation**: Implement comprehensive validation to catch errors early
4. **Environment Adaptation**: Automatically adjust settings based on runtime environment
5. **Experiment Management**: Organize and run complex experiments with proper tracking
6. **Testing**: Ensure configuration correctness with comprehensive testing

The power of configs-as-code allows you to build maintainable, scalable, and production-ready training pipelines that adapt to different environments and requirements while maintaining type safety and validation throughout. 