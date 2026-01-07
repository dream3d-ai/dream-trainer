# Tutorial: Multi-GPU Training

This tutorial teaches you how to scale your Dream Trainer from single-GPU to multi-GPU training using FSDP2 (Fully Sharded Data Parallel). You'll learn the different parallelism strategies and when to use each one.

## What You'll Learn

- How to configure FSDP2 for multi-GPU training
- Understanding data parallelism strategies
- Adding tensor parallelism for large models
- Multi-node training setup
- Performance optimization tips

## Prerequisites

- Completed the [First Trainer Tutorial](first-trainer.md)
- Multiple GPUs available (2+ recommended)
- Understanding of distributed training concepts

## Understanding Parallelism Strategies

Dream Trainer supports multiple parallelism strategies that can be combined:

| Strategy | What it Shards | When to Use |
|----------|---------------|-------------|
| **DDP** | Gradients only | Small models, fast communication |
| **FSDP2** | Parameters, gradients, optimizer states | Large models, memory-constrained |
| **Tensor Parallel (TP)** | Individual layers | Very large layers (e.g., LLM attention) |
| **Pipeline Parallel (PP)** | Model stages | Models too large for single GPU |
| **Context Parallel (CP)** | Sequence dimension | Extremely long sequences |

## Part 1: Basic FSDP2 Setup

### Step 1: Modify Your Configuration

The simplest way to enable multi-GPU training is to use FSDP2:

```python
from dataclasses import dataclass
from dream_trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin
from dream_trainer.configs import DeviceParameters, TrainingParameters


@dataclass
class MultiGPUConfig(BaseTrainerConfig, SetupConfigMixin):
    hidden_size: int = 1024
    num_layers: int = 12
    learning_rate: float = 3e-4


# Configure for FSDP2 across 4 GPUs
config = MultiGPUConfig(
    device_parameters=DeviceParameters.FSDP(
        dp_shard=4,  # Shard across 4 GPUs
    ),
    training_parameters=TrainingParameters(
        n_epochs=10,
        train_batch_size=32,  # Per-GPU batch size
    ),
)
```

### Step 2: Implement FSDP Sharding

Add the `apply_fully_shard` method to control how FSDP shards your model:

```python
from torch.distributed._composable.fsdp import fully_shard


class MultiGPUTrainer(BaseTrainer, SetupMixin):
    config: MultiGPUConfig

    def configure_models(self):
        self.model = TransformerModel(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
        )

    def apply_fully_shard(self, fsdp_config: dict):
        """Apply FSDP2 sharding to the model.

        This method is called automatically by SetupMixin when
        dp_shard > 1 in DeviceParameters.

        Args:
            fsdp_config: Dictionary with FSDP configuration including:
                - mesh: DeviceMesh for sharding
                - mp_policy: Mixed precision policy
                - reshard_after_forward: Whether to reshard after forward
        """
        # Strategy 1: Shard each transformer block
        # This is the most common approach for transformers
        for layer in self.model.layers:
            fully_shard(layer, **fsdp_config)

        # Shard the entire model (wraps remaining params)
        fully_shard(self.model, **fsdp_config)

    # ... other required methods
```

### Step 3: Launch Multi-GPU Training

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 train.py

# Or let Dream Trainer auto-detect GPUs
python train.py  # Will use all available GPUs
```

## Part 2: Understanding FSDP Sharding Strategies

### Per-Layer Sharding (Recommended for Transformers)

```python
def apply_fully_shard(self, fsdp_config: dict):
    """Shard each layer independently for best memory efficiency."""
    # Shard embedding layer
    fully_shard(self.model.embed_tokens, **fsdp_config)

    # Shard each transformer layer
    for layer in self.model.layers:
        fully_shard(layer, **fsdp_config)

    # Shard output projection
    fully_shard(self.model.lm_head, **fsdp_config)

    # Final wrap for any remaining parameters
    fully_shard(self.model, **fsdp_config)
```

### Block-Level Sharding (For Custom Architectures)

```python
def apply_fully_shard(self, fsdp_config: dict):
    """Shard attention and MLP blocks separately."""
    for layer in self.model.layers:
        # Shard attention block
        fully_shard(layer.attention, **fsdp_config)
        # Shard MLP block
        fully_shard(layer.mlp, **fsdp_config)

    fully_shard(self.model, **fsdp_config)
```

### Minimal Sharding (For Testing)

```python
def apply_fully_shard(self, fsdp_config: dict):
    """Shard only the entire model - simplest but less efficient."""
    fully_shard(self.model, **fsdp_config)
```

## Part 3: Adding Tensor Parallelism

For very large models, combine FSDP with Tensor Parallelism:

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


@dataclass
class LargeModelConfig(BaseTrainerConfig, SetupConfigMixin):
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32


config = LargeModelConfig(
    device_parameters=DeviceParameters(
        # 2-way tensor parallelism within nodes
        tensor_parallel=2,
        # 4-way FSDP across TP groups
        dp_shard=4,
    ),
    training_parameters=TrainingParameters(
        n_epochs=10,
        train_batch_size=8,
    ),
)


class LargeModelTrainer(BaseTrainer, SetupMixin):
    config: LargeModelConfig

    def apply_tensor_parallel(self, tp_mesh):
        """Apply tensor parallelism to attention layers.

        This is called BEFORE apply_fully_shard when both are enabled.

        Args:
            tp_mesh: DeviceMesh for tensor parallel group
        """
        for layer in self.model.layers:
            # Define tensor parallel plan for attention
            tp_plan = {
                # Query, Key, Value projections - split columns
                "attention.q_proj": ColwiseParallel(),
                "attention.k_proj": ColwiseParallel(),
                "attention.v_proj": ColwiseParallel(),
                # Output projection - split rows (to match column split)
                "attention.o_proj": RowwiseParallel(),

                # MLP layers
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            }

            parallelize_module(layer, tp_mesh, tp_plan)

    def apply_fully_shard(self, fsdp_config: dict):
        """Apply FSDP after tensor parallelism."""
        for layer in self.model.layers:
            fully_shard(layer, **fsdp_config)
        fully_shard(self.model, **fsdp_config)
```

### Understanding TP Plans

```python
# Tensor Parallel shards individual tensors:
#
# ColwiseParallel: Splits output features (columns)
#   [4096, 4096] -> GPU0: [4096, 2048], GPU1: [4096, 2048]
#
# RowwiseParallel: Splits input features (rows)
#   [4096, 4096] -> GPU0: [2048, 4096], GPU1: [2048, 4096]
#
# For attention:
#   QKV projections use ColwiseParallel (split heads)
#   Output projection uses RowwiseParallel (combine heads)
```

## Part 4: Multi-Node Training

### Configuration for Multiple Nodes

```python
config = LargeModelConfig(
    device_parameters=DeviceParameters(
        # HSDP: Tensor parallel within node, FSDP across nodes
        tensor_parallel=8,      # 8 GPUs per node for TP
        dp_shard=4,             # 4 nodes for FSDP
        dp_replicate=1,         # No replication
    ),
)
```

### Launch Script for Multi-Node

```bash
#!/bin/bash
# launch_multinode.sh

# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    train.py

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    train.py

# ... repeat for nodes 2 and 3
```

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=dream-trainer
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

# Get master address from first node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch training
srun torchrun \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py
```

## Part 5: Gradient Accumulation

For effective larger batch sizes without more memory:

```python
config = MultiGPUConfig(
    training_parameters=TrainingParameters(
        n_epochs=10,
        train_batch_size=8,             # Per-GPU micro-batch
        gradient_accumulation_steps=4,  # Accumulate 4 micro-batches
        # Effective batch = 8 * 4 * num_gpus
    ),
    device_parameters=DeviceParameters.FSDP(dp_shard=4),
)


class TrainerWithAccumulation(BaseTrainer, SetupMixin):
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        # backward() automatically scales loss for accumulation
        self.backward(loss)

        # step() is only called on accumulation boundaries
        if not self.is_accumulating_gradients:
            # Gradients are ready - step optimizer
            grad_norm = self.step(self.optimizer)
            return {"loss": loss, "grad_norm": grad_norm}

        return {"loss": loss}
```

## Part 6: Mixed Precision Training

Enable automatic mixed precision for faster training:

```python
import torch

config = MultiGPUConfig(
    device_parameters=DeviceParameters(
        dp_shard=4,
        # Mixed precision settings
        param_dtype=torch.bfloat16,  # Model weights in BF16
        reduce_dtype=torch.float32,   # Gradients reduced in FP32
    ),
)
```

### Using Loss Scaling (FP16)

```python
config = MultiGPUConfig(
    device_parameters=DeviceParameters(
        dp_shard=4,
        param_dtype=torch.float16,  # FP16 requires loss scaling
        reduce_dtype=torch.float32,
    ),
)


class FP16Trainer(BaseTrainer, SetupMixin):
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass (automatic autocast)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        # backward() handles loss scaling for FP16
        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            return {"loss": loss, "grad_norm": grad_norm}

        return {"loss": loss}
```

## Part 7: Performance Optimization

### Enable FSDP Prefetching

```python
from dream_trainer.callbacks import OptimizeFSDP

config = MultiGPUConfig(
    device_parameters=DeviceParameters.FSDP(dp_shard=4),
    callbacks=CallbackCollection([
        OptimizeFSDP(
            prefetch=2,  # Prefetch 2 layers ahead
            trace_execution=True,  # Auto-detect optimal order
        ),
    ]),
)
```

### Enable torch.compile

```python
config = MultiGPUConfig(
    device_parameters=DeviceParameters(
        dp_shard=4,
        compile_model=True,
        compiled_autograd=True,
    ),
)
```

### Optimize Communication

```bash
# Environment variables for optimal NCCL performance
export NCCL_IB_DISABLE=0           # Enable InfiniBand
export NCCL_IB_GID_INDEX=3         # Use correct GID
export NCCL_SOCKET_IFNAME=eth0     # Network interface
export NCCL_DEBUG=WARN             # Debug level

# For NVLink systems
export NCCL_P2P_LEVEL=NVL
```

## Part 8: Complete Multi-GPU Example

```python
"""Complete multi-GPU training example."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from dream_trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin
from dream_trainer.configs import (
    DeviceParameters,
    TrainingParameters,
    CheckpointParameters,
)
from dream_trainer.callbacks import (
    CheckpointCallback,
    ProgressBar,
    LoggerCallback,
    OptimizeFSDP,
    CallbackCollection,
)


@dataclass
class GPTConfig(BaseTrainerConfig, SetupConfigMixin):
    # Model
    vocab_size: int = 50257
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config.hidden_size, config.num_heads)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GPTTrainer(BaseTrainer, SetupMixin):
    config: GPTConfig

    def configure_models(self):
        self.model = GPTModel(self.config)

    def init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.model.apply(_init)

    def apply_tensor_parallel(self, tp_mesh):
        """Apply tensor parallelism to attention and MLP layers."""
        for layer in self.model.layers:
            tp_plan = {
                "attention": ColwiseParallel(input_layouts=None),
                "mlp.0": ColwiseParallel(),
                "mlp.2": RowwiseParallel(),
            }
            parallelize_module(layer, tp_mesh, tp_plan)

    def apply_fully_shard(self, fsdp_config):
        """Apply FSDP sharding."""
        fully_shard(self.model.embed_tokens, **fsdp_config)
        for layer in self.model.layers:
            fully_shard(layer, **fsdp_config)
        fully_shard(self.model, **fsdp_config)

    def configure_optimizers(self):
        # Separate weight decay groups
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.config.learning_rate)

    def configure_dataloaders(self):
        # Synthetic data for demo
        train_data = torch.randint(0, self.config.vocab_size, (10000, 512))
        val_data = torch.randint(0, self.config.vocab_size, (1000, 512))

        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=self.config.training_parameters.train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=self.config.training_parameters.train_batch_size * 2,
        )
        return train_loader, val_loader

    def training_step(self, batch, batch_idx):
        (input_ids,) = batch
        input_ids = input_ids.to(self.device)

        # Shift for next-token prediction
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        # Forward
        with self.loss_parallel():
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        self.backward(loss)

        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.optimizer)
            return {"loss": loss, "grad_norm": grad_norm}

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        (input_ids,) = batch
        input_ids = input_ids.to(self.device)

        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
        )

        return {"val_loss": loss}


def main():
    callbacks = CallbackCollection([
        ProgressBar(),
        LoggerCallback(log_every_n_train_batches=10),
        OptimizeFSDP(prefetch=2),
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_dir="./checkpoints",
                checkpoint_every_n_epochs=1,
            )
        ),
    ])

    config = GPTConfig(
        device_parameters=DeviceParameters(
            tensor_parallel=2,  # 2-way TP
            dp_shard=4,         # 4-way FSDP
            param_dtype=torch.bfloat16,
            compile_model=True,
        ),
        training_parameters=TrainingParameters(
            n_epochs=10,
            train_batch_size=8,
            gradient_accumulation_steps=4,
            gradient_clip_val=1.0,
            val_frequency=0.25,
        ),
        callbacks=callbacks,
    )

    trainer = GPTTrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce per-GPU batch size
training_parameters=TrainingParameters(
    train_batch_size=4,  # Smaller batch
    gradient_accumulation_steps=8,  # More accumulation
)

# Enable CPU offloading
device_parameters=DeviceParameters(
    dp_shard=4,
    cpu_offload=True,  # Offload optimizer states to CPU
)

# Enable activation checkpointing
device_parameters=DeviceParameters(
    dp_shard=4,
    checkpoint_activations=True,
)
```

### Slow Communication

```bash
# Check NCCL topology
nvidia-smi topo -m

# Use optimal NCCL settings
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
```

### Hangs on Initialization

```bash
# Debug distributed initialization
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# Check firewall
sudo ufw allow 29500
```

## Next Steps

- [Custom Components](custom-components.md): Create your own mixins
- [Production Setup](production.md): Deploy to production clusters
- [Parallelism Guide](../parallelism.md): Deep dive into parallelism strategies
- [Performance Guide](../performance.md): Optimize training throughput
