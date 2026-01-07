# Frequently Asked Questions

## General Questions

### What is Dream Trainer?

Dream Trainer is a composable distributed training framework built exclusively around PyTorch's DTensor abstractions. It provides a flexible, mixin-based architecture that makes it easy to adopt the latest PyTorch distributed patterns while keeping your training code clean and debuggable.

### How is Dream Trainer different from PyTorch Lightning?

| Feature | Dream Trainer | PyTorch Lightning |
|---------|---------------|-------------------|
| **Architecture** | Mixin-based composition | Inheritance-based |
| **Distributed backend** | DTensor-native | DDP/FSDP wrappers |
| **Configuration** | Python dataclasses | YAML/config files |
| **Parallelism** | FSDP2, TP, PP, CP first-class | Plugin-based |
| **Philosophy** | Explicit, minimal magic | Comprehensive, batteries-included |

Dream Trainer is ideal if you want direct control over distributed training with modern PyTorch APIs.

### What PyTorch version is required?

Dream Trainer requires PyTorch >= 2.7.0 for full DTensor support. Some features may work with earlier versions, but FSDP2 and tensor parallelism require 2.7+.

### Does Dream Trainer support CPU-only training?

Yes, but Dream Trainer is optimized for GPU training. CPU training works for development and testing but is not recommended for production workloads.

---

## Training Questions

### How do I use multiple GPUs?

```python
from dream_trainer.configs import DeviceParameters

# FSDP across 4 GPUs
config = MyConfig(
    device_parameters=DeviceParameters.FSDP(dp_shard=4),
)

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

### How do I enable mixed precision?

```python
import torch
from dream_trainer.configs import DeviceParameters

config = MyConfig(
    device_parameters=DeviceParameters(
        param_dtype=torch.bfloat16,  # or torch.float16
        reduce_dtype=torch.float32,
    ),
)
```

### How do I save and load checkpoints?

```python
from dream_trainer.callbacks import CheckpointCallback
from dream_trainer.configs import CheckpointParameters

config = MyConfig(
    callbacks=CallbackCollection([
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_dir="./checkpoints",
                checkpoint_every_n_epochs=1,
                keep_top_k=3,
            )
        ),
    ]),
)

# To resume from checkpoint:
trainer = MyTrainer(config)
trainer.load_from_checkpoint("./checkpoints/checkpoint_epoch_5")
trainer.fit()
```

### How do I use gradient accumulation?

```python
from dream_trainer.configs import TrainingParameters

config = MyConfig(
    training_parameters=TrainingParameters(
        train_batch_size=4,              # Per-GPU micro-batch
        gradient_accumulation_steps=8,   # Effective batch = 4 * 8 * num_gpus
    ),
)
```

### How do I add learning rate warmup?

```python
class MyTrainer(DreamTrainer):
    def configure_schedulers(self):
        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 0.5 * (1 + math.cos(
                math.pi * (step - self.config.warmup_steps) /
                (self.config.max_steps - self.config.warmup_steps)
            ))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
```

---

## Architecture Questions

### What methods must I implement in a trainer?

At minimum, with `BaseTrainer` + `SetupMixin`:

```python
class MyTrainer(BaseTrainer, SetupMixin):
    # Required by SetupMixin:
    def configure_models(self): ...
    def init_weights(self): ...
    def configure_optimizers(self): ...
    def configure_dataloaders(self): ...

    # Required by BaseTrainer:
    def training_step(self, batch, batch_idx): ...
    def validation_step(self, batch, batch_idx): ...
```

### What mixins are available?

| Mixin | Purpose |
|-------|---------|
| `SetupMixin` | Model, optimizer, dataloader setup |
| `EvalMetricMixin` | torchmetrics integration |
| `WandBLoggerMixin` | Weights & Biases logging |
| `QuantizeMixin` | FP8 quantization |
| `LoggerMixin` | Generic logging interface |

### How do I create a custom mixin?

```python
from dataclasses import dataclass

@dataclass
class MyMixinConfig:
    my_setting: float = 1.0

class MyMixin:
    config: MyMixinConfig

    def setup(self):
        super().setup()
        self._my_setup()

    def _my_setup(self):
        # Custom initialization
        pass
```

### How do callbacks differ from mixins?

**Callbacks**: Hook into the training lifecycle without modifying the trainer. Good for side effects like logging, checkpointing, early stopping.

**Mixins**: Add new methods and state to the trainer. Good for core functionality like model setup, metrics tracking, different logging backends.

---

## Distributed Questions

### How do I train on multiple nodes?

```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=<NODE0_IP> --master_port=29500 train.py

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=<NODE0_IP> --master_port=29500 train.py
```

### How do I use tensor parallelism?

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        tensor_parallel=2,  # 2-way tensor parallelism
        dp_shard=4,         # Combined with 4-way FSDP
    ),
)

class MyTrainer(DreamTrainer):
    def apply_tensor_parallel(self, tp_mesh):
        for layer in self.model.layers:
            parallelize_module(layer, tp_mesh, {
                "attn.qkv": ColwiseParallel(),
                "attn.out": RowwiseParallel(),
            })
```

### How do I debug distributed training?

```python
# Enable debug mode
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"

# Use the debug callback
from dream_trainer.callbacks import DebugCallback
config = MyConfig(
    callbacks=CallbackCollection([DebugCallback()]),
)
```

---

## Performance Questions

### Why is my training slow?

Common causes:
1. **Data loading bottleneck**: Increase `num_workers`, use `pin_memory=True`
2. **Missing mixed precision**: Enable `param_dtype=torch.bfloat16`
3. **Missing compilation**: Enable `compile_model=True`
4. **Poor batch size**: Find optimal with memory profiling
5. **Communication overhead**: Enable FSDP prefetching

### How do I profile training?

```python
from dream_trainer.callbacks import ProfileCallback

config = MyConfig(
    callbacks=CallbackCollection([
        ProfileCallback(
            profile_steps=range(10, 20),
            output_dir="./profiles",
        ),
    ]),
)
```

### How do I reduce memory usage?

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        checkpoint_activations=True,  # Activation checkpointing
        cpu_offload=True,             # Offload optimizer states
    ),
    training_parameters=TrainingParameters(
        train_batch_size=2,           # Smaller batch
        gradient_accumulation_steps=16,  # Compensate with accumulation
    ),
)
```

---

## Configuration Questions

### How do I compose configurations?

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_layers: int = 12

@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    warmup_steps: int = 1000

@dataclass
class MyConfig(BaseTrainerConfig, ModelConfig, TrainingConfig):
    # Combines all fields
    pass
```

### How do I use environment variables in config?

```python
import os
from dataclasses import dataclass, field

@dataclass
class MyConfig(DreamTrainerConfig):
    data_path: str = field(
        default_factory=lambda: os.environ.get("DATA_PATH", "/data")
    )
    num_workers: int = field(
        default_factory=lambda: int(os.environ.get("NUM_WORKERS", "4"))
    )
```

### How do I validate configuration?

```python
@dataclass
class MyConfig(DreamTrainerConfig):
    learning_rate: float = 3e-4
    num_layers: int = 12

    def validate(self):
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_layers >= 1, "Need at least 1 layer"
```

---

## Troubleshooting

### "CUDA out of memory"

See the [memory optimization section](performance.md#memory-optimization) in the Performance Guide.

### Training hangs

1. Check all ranks reach the same point (add print statements with rank)
2. Enable NCCL debug: `export NCCL_DEBUG=INFO`
3. Check network connectivity between nodes
4. Verify firewall allows port 29500

### "RuntimeError: Expected all tensors on same device"

Ensure all tensors are moved to the correct device:
```python
def training_step(self, batch, batch_idx):
    x = batch['input'].to(self.device)  # Use self.device
    y = batch['target'].to(self.device)
```

### Results differ between runs

Enable deterministic mode:
```python
from dream_trainer.utils import seed_everything
seed_everything(42, deterministic=True)
```

---

## More Resources

- [Installation Guide](installation.md)
- [Getting Started](getting-started.md)
- [Debugging Guide](debugging.md)
- [Performance Guide](performance.md)
- [Troubleshooting](troubleshooting.md)
