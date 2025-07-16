# Core Concepts

Welcome to the Dream Trainer Core Concepts guide. This document explains the fundamental ideas and design patterns that power Dream Trainer's flexible, distributed training framework.

## Understanding DTensor

### What is DTensor?

DTensor (Distributed Tensor) is PyTorch's next-generation distributed computing primitive that provides a unified abstraction for tensor parallelism. Unlike traditional distributed approaches that treat each device's data separately, DTensor represents a global logical tensor that can be sharded across multiple devices while maintaining a single, coherent view.

**Key Benefits:**
- **Unified API**: Write code once, run with any parallelism strategy
- **Automatic gradient synchronization**: DTensor handles communication patterns
- **Composability**: Easily combine different parallelism strategies
- **Device mesh awareness**: Understands multi-dimensional device topologies

### DTensor vs Traditional Distributed Training

```python
# Traditional DDP approach - each rank has its own tensor
model = MyModel().to(rank)
model = DDP(model, device_ids=[rank])
output = model(input)  # Each rank processes different data

# DTensor approach - single logical tensor across all ranks
from torch.distributed.tensor import DTensor, Shard, Replicate, DeviceMesh

# Create a 2D device mesh for hybrid parallelism
device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])  # 2x2 mesh

# Create a DTensor with specific sharding
tensor = DTensor.from_local(
    local_tensor,
    device_mesh,
    placements=[Shard(0), Replicate()]  # Shard on dim 0, replicate on dim 1
)
```

### Placement and Sharding

DTensor uses **placements** to describe how data is distributed:

- **`Replicate()`**: Each device has a full copy of the tensor
- **`Shard(dim)`**: Tensor is split along the specified dimension
- **`Partial()`**: Each device has a partial value (used during reduction)

```python
# Example: Creating different DTensor distributions
import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate

# Initialize a simple device mesh
device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])

# Full replication - all devices have complete tensor
weight_replicated = DTensor.from_local(
    torch.randn(1024, 1024),
    device_mesh,
    [Replicate()]
)

# Sharding along dimension 0 - each device gets 256 rows
weight_sharded = DTensor.from_local(
    torch.randn(256, 1024),  # Local shard size
    device_mesh,
    [Shard(0)]
)

# 2D mesh with hybrid sharding
mesh_2d = DeviceMesh("cuda", [[0, 1], [2, 3]])
weight_2d = DTensor.from_local(
    torch.randn(512, 512),
    mesh_2d,
    [Shard(0), Shard(1)]  # Shard rows on first mesh dim, cols on second
)
```

### DeviceMesh Concept

DeviceMesh represents the topology of devices participating in distributed computation:

```python
# 1D mesh - simple data parallel
dp_mesh = DeviceMesh("cuda", [0, 1, 2, 3])

# 2D mesh - hybrid data + tensor parallel
# First dimension: data parallel groups
# Second dimension: tensor parallel groups
mesh_2d = DeviceMesh("cuda", [
    [0, 1, 2, 3],  # TP group 0
    [4, 5, 6, 7],  # TP group 1
])

# 3D mesh - data + tensor + pipeline parallel
mesh_3d = DeviceMesh("cuda", [
    [[0, 1], [2, 3]],    # PP stage 0
    [[4, 5], [6, 7]],    # PP stage 1
])
```

## The Mixin Architecture

### Why Mixins?

Dream Trainer uses a mixin-based architecture to achieve maximum flexibility and code reuse. Instead of a monolithic trainer class with every feature built-in, we compose trainers from smaller, focused components.

**Traditional Inheritance Problems:**
```python
# Rigid hierarchy - hard to mix features
class Trainer:
    def train(self): ...

class DistributedTrainer(Trainer):
    def setup_distributed(self): ...

class LoggingTrainer(DistributedTrainer):
    def log_metrics(self): ...

# What if I want logging without distributed? 🤔
```

**Mixin Solution:**
```python
# Flexible composition - take what you need
class MyTrainer(BaseTrainer, SetupMixin, LoggerMixin):
    # Get distributed setup from SetupMixin
    # Get logging from LoggerMixin
    # Skip what you don't need!
```

### How Mixins Work

Mixins leverage Python's Method Resolution Order (MRO) to combine functionality:

```python
# Understanding MRO with mixins
class BaseTrainer:
    def setup(self):
        print("BaseTrainer.setup()")

class ModelMixin:
    def setup(self):
        super().setup()  # Calls next in MRO
        print("ModelMixin.setup()")
        self.setup_models()

class OptimizerMixin:
    def setup(self):
        super().setup()  # Calls next in MRO
        print("OptimizerMixin.setup()")
        self.setup_optimizers()

class MyTrainer(BaseTrainer, ModelMixin, OptimizerMixin):
    def setup(self):
        super().setup()  # Triggers the chain
        print("MyTrainer.setup()")

# MRO: MyTrainer -> OptimizerMixin -> ModelMixin -> BaseTrainer
# Output:
# BaseTrainer.setup()
# ModelMixin.setup()
# OptimizerMixin.setup()
# MyTrainer.setup()
```

### Available Mixins

Dream Trainer provides several mixins for different aspects of training:

#### SetupMixin Family
Handles model initialization, parallelism, and optimization:

```python
from dream_trainer.trainer.mixins import SetupMixin

class MyTrainer(BaseTrainer, SetupMixin):
    def configure_models(self):
        # Create models on meta device for efficiency
        self.model = TransformerModel(self.config)
    
    def init_weights(self):
        # Initialize weights after device placement
        self.model.apply(self._init_weights)
    
    def configure_optimizers(self):
        # Create optimizers after model is on device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def configure_dataloaders(self):
        # Return train and validation dataloaders
        return train_dataloader, val_dataloader
```

The SetupMixin is actually composed of several sub-mixins:
- **ModelSetupMixin**: Model creation, parallelism application, compilation
- **OptimizerAndSchedulerSetupMixin**: Optimizer and LR scheduler management  
- **DataLoaderSetupMixin**: DataLoader configuration and setup

#### EvalMetricMixin
Integrates torchmetrics for standardized evaluation:

```python
from dream_trainer.trainer.mixins import EvalMetricMixin

class MyTrainer(BaseTrainer, SetupMixin, EvalMetricMixin):
    def configure_metrics(self):
        # Metrics are automatically moved to correct devices
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=10)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        # Update metrics - handles distributed sync automatically
        self.accuracy(logits, y)
        self.f1_score(logits, y)
        
        return {"val_loss": F.cross_entropy(logits, y)}
```

#### LoggerMixin Variants
Different logging backends with consistent interfaces:

```python
from dream_trainer.trainer.mixins import WandBLoggerMixin

class MyTrainer(BaseTrainer, SetupMixin, WandBLoggerMixin):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        
        # Automatic logging to Weights & Biases
        self.log_scalar("train/loss", loss)
        
        if batch_idx % 100 == 0:
            self.log_image("train/samples", batch["image"][:8])
        
        return {"loss": loss}
```

#### QuantizeMixin
For model quantization (FP8, INT8):

```python
from dream_trainer.trainer.mixins import QuantizeMixin

class MyTrainer(BaseTrainer, SetupMixin, QuantizeMixin):
    def setup(self):
        super().setup()
        # Quantize after model setup
        self.apply_quantization()
```

### Building a Custom Trainer

Here's a complete example showing how to compose a custom trainer:

```python
from dataclasses import dataclass
from dream_trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.trainer.mixins import (
    SetupMixin, SetupConfigMixin,
    EvalMetricMixin, EvalMetricConfigMixin,
    WandBLoggerMixin, WandBLoggerConfigMixin
)

# Step 1: Define configuration by mixing config classes
@dataclass(kw_only=True)
class MyTrainerConfig(
    BaseTrainerConfig,
    SetupConfigMixin,
    EvalMetricConfigMixin,
    WandBLoggerConfigMixin
):
    # Add custom config fields
    model_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    vocab_size: int = 50257

# Step 2: Define trainer by mixing trainer classes
class MyTrainer(
    BaseTrainer,
    SetupMixin,
    EvalMetricMixin,
    WandBLoggerMixin
):
    config: MyTrainerConfig
    
    def configure_models(self):
        self.model = TransformerLM(
            dim=self.config.model_dim,
            heads=self.config.num_heads,
            layers=self.config.num_layers,
            vocab_size=self.config.vocab_size
        )
    
    def init_weights(self):
        # Custom weight initialization
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        
        self.model.apply(_init_weights)
    
    def configure_optimizers(self):
        # Separate weight decay for different param groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=self.config.learning_rate)
    
    def configure_schedulers(self):
        # Cosine annealing with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training_parameters.epochs
        )
    
    def configure_dataloaders(self):
        train_dataset = MyDataset("train", self.config)
        val_dataset = MyDataset("validation", self.config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def configure_metrics(self):
        self.perplexity = torchmetrics.Perplexity()
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Forward pass
        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass with gradient accumulation
        self.backward(loss)
        
        # Step optimizer when not accumulating
        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.model, self.optimizer)
            self.log_scalar("train/grad_norm", grad_norm)
        
        # Log metrics
        self.log_scalar("train/loss", loss)
        self.log_scalar("train/lr", self.optimizer.param_groups[0]["lr"])
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Update metrics
        self.perplexity(logits, labels)
        
        return {"val_loss": loss}
```

## Training Loop Lifecycle

The training loop in Dream Trainer follows a well-defined lifecycle with clear phases and hooks for customization:

### Initialization Phase

```python
trainer = MyTrainer(config)
# 1. __init__ is called
#    - Initialize world/distributed setup
#    - Create callback collection
#    - Set initial state (epoch=0, global_step=0)

trainer.fit()
# 2. configure() is called
#    - configure_models() on meta device
#    - post_configure_models() hook

# 3. setup() is called
#    - Apply parallelism (TP, PP, FSDP)
#    - Initialize weights
#    - Setup optimizers & schedulers
#    - Setup dataloaders
#    - Setup metrics
```

### Training Phase

The training loop executes these steps for each epoch:

```python
def perform_training_epoch(self):
    self.train()  # Set models to training mode
    self.callbacks.pre_train_epoch(self)
    
    for batch_idx, batch in enumerate(self.train_dataloader):
        # 1. Pre-batch callbacks
        self.callbacks.pre_train_batch(self, batch, batch_idx)
        
        # 2. Training step with autocast
        with self.train_context():  # Includes autocast, profiling, etc.
            output = self.training_step(batch, batch_idx)
        
        # 3. Post-batch callbacks
        self.callbacks.post_train_batch(self, output, batch, batch_idx)
        
        # 4. Increment counters
        self.local_batches += 1
        if not self.is_accumulating_gradients:
            self.global_step += 1
        
        # 5. Validation check
        if should_validate():
            self.perform_validation_epoch()
    
    self.callbacks.post_train_epoch(self)
```

### Gradient Accumulation Flow

Dream Trainer handles gradient accumulation automatically:

```python
def training_step(self, batch, batch_idx):
    # Your implementation
    loss = self.model(batch)
    
    # backward() handles scaling by accumulation steps
    self.backward(loss)  # Internally: (loss / accumulation_steps).backward()
    
    # Only step optimizer when gradients are ready
    if not self.is_accumulating_gradients:
        # All gradients accumulated, time to step
        grad_norm = self.step(self.model, self.optimizer)
    
    return {"loss": loss}
```

The `no_gradient_sync` context manager optimizes distributed training:

```python
# Automatically applied during accumulation steps
with self.no_gradient_sync(self.model):
    loss.backward()  # No distributed sync until final accumulation
```

### Validation Phase

Validation runs with gradients disabled and models in eval mode:

```python
@torch.no_grad()
def perform_validation_epoch(self):
    self.eval()  # Set models to eval mode
    self.callbacks.pre_val_epoch(self)
    
    for batch_idx, batch in enumerate(self.val_dataloader):
        # 1. Pre-batch callbacks
        self.callbacks.pre_val_batch(self, batch, batch_idx)
        
        # 2. Validation step
        output = self.validation_step(batch, batch_idx)
        
        # 3. Post-batch callbacks
        self.callbacks.post_val_batch(self, output, batch, batch_idx)
    
    # 4. Compute epoch metrics
    if hasattr(self, "compute_metrics"):
        metrics = self.compute_metrics()
        self.callbacks.on_val_epoch_end(self, metrics)
    
    self.callbacks.post_val_epoch(self)
```

### Callback Integration

Callbacks provide hooks at every stage of training:

```python
from dream_trainer.callbacks import Callback

class MyCallback(Callback):
    def on_train_start(self, trainer):
        print("Training started!")
    
    def on_train_batch_end(self, trainer, output, batch, batch_idx):
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: loss = {output['loss']:.4f}")
    
    def on_validation_end(self, trainer, metrics):
        print(f"Validation perplexity: {metrics['perplexity']:.2f}")
```

## State Management

Dream Trainer provides comprehensive state management for checkpointing and resumption:

### Trainer State

The trainer maintains several state variables:

```python
class BaseTrainer:
    # Global state
    global_step: int = 0        # Total optimizer steps across all epochs
    current_epoch: int = 0      # Current epoch number
    local_batches: int = 0      # Batches processed since start
    
    # Training flags
    training: bool = False      # True during training, False during eval
    
    # Properties
    @property
    def is_accumulating_gradients(self) -> bool:
        """True if currently accumulating gradients"""
        return (
            (self.local_batches + 1) % self._num_gradient_accumulation_steps != 0
        ) and not self._is_last_training_batch
```

### Checkpointing

State dict includes all components needed for exact resumption:

```python
def state_dict(self) -> dict[str, Any]:
    return {
        "trainer": {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "callbacks": self.callbacks.state_dict(),
        },
        "models": {
            name: model.state_dict() 
            for name, model in self.named_models().items()
        },
        "optimizers": {
            name: optimizer.state_dict()
            for name, optimizer in self.named_optimizers().items()
        },
        "schedulers": {
            name: scheduler.state_dict()
            for name, scheduler in (self.named_schedulers() or {}).items()
        },
        "dataloaders": {
            "train": getattr(self.train_dataloader, "state_dict", lambda: {})(),
            "val": getattr(self.val_dataloader, "state_dict", lambda: {})(),
        },
    }

def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
    # Restore all components
    trainer_state = state_dict["trainer"]
    self.global_step = trainer_state["global_step"]
    self.current_epoch = trainer_state["current_epoch"]
    self.callbacks.load_state_dict(trainer_state["callbacks"])
    
    # Load model, optimizer, scheduler states...
```

### Distributed State

The `DistributedWorld` class manages distributed training state:

```python
class DistributedWorld:
    def __init__(self, device_parameters: DeviceParameters):
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.local_rank = torch.distributed.get_local_rank()
        
        # Device meshes for different parallelism types
        self.dp_mesh = self._build_mesh("dp", device_parameters)
        self.tp_mesh = self._build_mesh("tp", device_parameters)
        self.pp_mesh = self._build_mesh("pp", device_parameters)
    
    @contextlib.contextmanager
    def train_context(self):
        """Context for training with autocast and other optimizations"""
        with torch.cuda.amp.autocast(
            enabled=self.mixed_precision_enabled,
            dtype=self.mixed_precision_dtype
        ):
            yield
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce across data parallel group"""
        return dist_ops.all_reduce(tensor, group=self.dp_process_group)
    
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather across data parallel group"""
        return dist_ops.all_gather_tensor(tensor, group=self.dp_process_group)
```

### Fault Tolerance

Dream Trainer integrates with PyTorch's fault tolerance mechanisms:

```python
# Automatic checkpoint saving
checkpoint_callback = CheckpointCallback(
    checkpoint_dir="./checkpoints",
    save_frequency=1000,  # Save every 1000 steps
    keep_last_n=3,        # Keep last 3 checkpoints
)

# Fault-tolerant training with torchft
if config.enable_fault_tolerance:
    from torchft import FaultTolerantWorld
    trainer.world = FaultTolerantWorld(trainer.world)
```

### Memory Management

DTensor enables efficient memory usage through sharding:

```python
# Memory usage comparison
# Traditional: Each GPU stores full 405B parameter model = 810GB per GPU
# With FSDP sharding: 810GB / 8 GPUs = ~101GB per GPU

# Configure memory-efficient training
config = DreamTrainerConfig(
    device_parameters=DeviceParameters(
        dp_degree=8,  # Shard across 8 GPUs
        enable_fsdp=True,
        fsdp_limit_all_gathers=True,  # Rate limit to save memory
        fsdp_forward_prefetch=True,   # Overlap computation/communication
    )
)
```

## Next Steps

Now that you understand the core concepts:

1. **[Getting Started](getting-started.md)**: Install Dream Trainer and run your first training
2. **[Configuration Guide](configuration.md)**: Deep dive into configuration options
3. **[Trainer Guide](trainer-guide.md)**: Build custom trainers for your use case
4. **[Callbacks](callbacks.md)**: Extend functionality with the callback system
5. **[API Reference](api/index.md)**: Detailed API documentation

## References

- [PyTorch DTensor RFC](https://github.com/pytorch/pytorch/issues/88838)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [TorchTitan: Production-Ready LLM Training](https://arxiv.org/html/2410.06511v1) 