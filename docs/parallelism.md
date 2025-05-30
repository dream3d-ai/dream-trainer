# Parallelism Guide

This guide explains the different types of parallelism supported by Dream Trainer and how to use them effectively.

For a complete example see `dream-trainer/examples/llama3/setup.py`

## Table of Contents

- [Overview](#overview)
- [Data Parallelism](#data-parallelism)
- [FSDP](#fsdp)
- [Tensor Parallelism](#tensor-parallelism)
- [Context Parallelism](#context-parallelism)
- [Pipeline Parallelism](#pipeline-parallelism)
- [Combining Parallelism](#combining-parallelism)

## Overview

Dream Trainer is built around PyTorch's DTensor abstractions, providing a unified interface for all parallelism schemes. Each type of parallelism serves a specific purpose:

- **Data Parallelism**: Scale training across multiple GPUs by replicating the model
- **FSDP2**: Second-generation Fully-Sharded Data Parallel built on DTensor
- **Tensor Parallelism**: Split model parameters across GPUs for larger models
- **Context Parallelism**: Handle long sequences by splitting across GPUs
- **Pipeline Parallelism**: Split model layers across GPUs for efficient memory usage

## Data Parallelism

Data Parallelism is the simplest form of parallelism, where the model is replicated across GPUs and each GPU processes a different batch of data.

### Using PyTorch's Replicate API

```python
from dream_trainer.trainer.mixins import ModelSetupMixin
from dream_trainer.configs import DeviceParameters
from torch.distributed.device_mesh import DeviceMesh

config = DreamTrainerConfig(
    device_parameters=DeviceParameters.DDP(
        compile_model=True,
        checkpoint_activations=False,
    )
)

class MyTrainer(ModelSetupMixin):
    def apply_replicate(self, dp_replicate_mesh: DeviceMesh):
        # Wraps the model in place with Distributed Data Parallel
        replicate(self.model, device_mesh=dp_replicate_mesh, bucket_cap_mb=100)
```

### Key Features

- **Simple Setup**: Just specify the number of GPUs
- **Linear Scaling**: Training speed scales linearly with GPU count
- **Memory Efficient**: Each GPU holds a complete model copy
- **Gradient Synchronization**: Automatic gradient averaging across GPUs

### Best Practices

1. Use when:

   - Model fits in GPU memory
   - Batch size can be increased
   - Training speed is the priority

2. Avoid when:
   - Model is too large for single GPU
   - Memory efficiency is critical
   - Need more advanced parallelism

## FSDP

Fully Sharded Data Parallel (FSDP) reduces memory usage by sharding model parameters across GPUs.

### Using FSDP2 API

```python
from dream_trainer import DreamTrainerConfig
from dream_trainer.configs import DeviceParameters

config = DreamTrainerConfig(
    device_parameters=DeviceParameters.FSDP(
        tensor_parallel=1,
        dp_shard="auto",
        compile_model=True,
        cpu_offload=False,
        checkpoint_activations=False,
    )
)

class MyTrainer(ModelSetupMixin):
    def apply_fully_shard(self, config: dict[str, Any]) -> None:
        # NOTE: if using Pipeline Parallelism, make sure to set reshard_after_forward=False on all layers for optimal performance.
        for layer in self.model.layers:
            fully_shard(layer, **config)
        fully_shard(self.model, **config, reshard_after_forward=False)
```

Alternatively, we can define our sharding strategy directly on the model with `fsdp2-utils` for simpler usage. `apply_fully_shard` will recurisvely call `Model.fully_shard` on all of the model's submodules that conform to the `FullyShard` protocol.

> **Note:**  
> All of the model's layers need to be wrapped with `fully_shard` to ensure inputs & layers are properly casted to the correct dtype/device specified by the `MixedPrecisionPolicy`. This casting & device movement is handled internally by FSDP.

```python
import torch.nn as nn
from typing import Any
from fsdp2_utils import apply_fully_shard, FullyShard

class TransformerBlock(nn.Module, FullyShard):
    attention: Attention
    feed_forward: FeedForward
    attention_norm: nn.RMSNorm
    ffn_norm: nn.RMSNorm

    def fully_shard(self, config: dict[str, Any]):
        fully_shard(self.attention, **config)
        fully_shard(self.feed_forward, **config)
        fully_shard(self.attention_norm, **config)
        fully_shard(self.ffn_norm, **config)

class Transformer(nn.Module, FullyShard):
    input: nn.Linear
    layers: nn.ModuleList
    output: nn.Linear

    def fully_shard(self, config: dict[str, Any]):
        fully_shard(self.input, **config)
        fully_shard(self.layers, **config)
        fully_shard(self.output, **config, reshard_after_forward=False)

class MyTrainer(ModelSetupMixin):
    def apply_fully_shard(self, config: dict[str, Any]) -> None:
        # apply_fully_shard will override reshard_after_forward to `False` for all blocks when Pipeline Parallelism is enabled.
        apply_fully_shard(self.model, config, pp_enabled=self.world.pp_enabled)

```

### Key Features

- **Memory Efficiency**: Parameters are sharded across GPUs
- **Mixed Precision**: Native support for FP16/BF16 mixed precision training
- **Gradient Sharding**: Reduces memory during backward pass

### Best Practices

1. Use when:

   - Model is too large for single GPU
   - Memory efficiency is important
   - Training speed can be sacrificed for memory

2. Configuration Tips:
   - Choose sharding strategy based on memory constraints
   - Enable mixed precision for better performance
   - Use activation checkpointing for very large models

[Read more about Fully Sharded Data Parallel (FSDP) in the PyTorch documentation.](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)

## Tensor Parallelism

Tensor Parallelism splits model parameters across GPUs, allowing for even larger models. Again, we'll use `fsdp2-utils` to simplify how we apply tensor parallelism.

### Configuration

```python
import torch.nn as nn
from typing import Any
from fsdp2_utils import apply_tensor_parallel, ParallelPlan

from dream_trainer import DreamTrainerConfig
from dream_trainer.configs import DeviceParameters

config = DreamTrainerConfig(
    device_parameters=DeviceParameters.FSDP(
        tensor_parallel="auto",
        dp_shard=1,    # no FSDP
        compile_model=True,
        cpu_offload=False,
        checkpoint_activations=False,
    )
)

class TransformerBlockParallel(ParallelPlan):
    attention_norm: nn.RMSNorm
    attention: "Attention"
    feed_forward: "FeedForward"
    ffn_norm: nn.RMSNorm

    def parallelize_plan(self, _):
        return {
            "attention_norm": sequence_parallel(self.attention_norm),
            "attention": prepare_module_input(
                self.attention,
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(self.attention.wq),
            "attention.wk": colwise_parallel(self.attention.wk),
            "attention.wv": colwise_parallel(self.attention.wv),
            "attention.wo": rowwise_parallel(self.attention.wo, output_layouts=Shard(1)),
            "ffn_norm": sequence_parallel(self.ffn_norm),
            "feed_forward": prepare_module_input(
                self.feed_forward,
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(self.feed_forward.w1),
            "feed_forward.w2": rowwise_parallel(self.feed_forward.w2, output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(self.feed_forward.w3),
        }


class TransformerParallel(FullyShard, ParallelPlan):
    tok_embeddings: nn.Embedding
    norm: nn.RMSNorm
    output: nn.Linear
    layers: nn.ModuleDict

    def parallelize_plan(self, loss_parallel: bool):
        return (
            {
                "tok_embeddings": rowwise_parallel(
                    self.tok_embeddings,
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "norm": sequence_parallel(self.norm),
                "output": colwise_parallel(
                    self.output,
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1) if loss_parallel else Replicate(),
                    use_local_output=not loss_parallel,
                ),
            },
        )

class MyTrainer(ModelSetupMixin):
    def apply_fully_shard(self, tp_mesh: DeviceMesh) -> None:
        apply_tensor_parallel(self.model, tp_mesh=tp_mesh, loss_parallel=self.world.loss_parallel_enabled)

```

> **Note:**  
> Using `fsdp2-utils` greatly simplifies the construction of a parallel plan.
>
> Without `fsdp2-utils`, you would need to manually build a parallel plan using PyTorch's classes like `ColwiseParallel` or `RowwiseParallel` for each layer. This process can become complex, especially if you want to use features like fp8 quantization, which would require using `Fp8ColwiseParallel` or similar classes for the affected layers.
>
> With `fsdp2-utils`, you only need to define a `parallel_plan` function for your model or block. The utility will automatically generate the correct plan at runtime, choosing the appropriate parallelization strategy (including fp8 support) for each layer.

### Key Features

- **Parameter Sharding**: Split large tensors across GPUs
- **Communication Efficiency**: Minimizes cross-GPU communication
- **Flexible Sharding**: Choose which dimensions to split

### Best Practices

1. Use when:

   - Model has large parameter tensors
   - Need more memory efficiency than FSDP
   - Want to combine with other parallelism

2. Configuration Tips:
   - Choose parallel dimension carefully
   - Consider communication overhead
   - Use with FSDP for maximum memory efficiency

## Context Parallelism

Context Parallelism splits sequences across GPUs, useful for long-context models.

### Configuration

```python
from dream_trainer import DreamTrainerConfig
from dream_trainer.configs import DeviceParameters

config = DreamTrainerConfig(
    device_parameters=DeviceParameters.FSDP(
        tensor_parallel=2,    # Split across 2 GPUs
        dp_shard="auto",
    )
)
```

### Key Features

- **Sequence Splitting**: Distribute long sequences across GPUs
- **Efficient Attention**: Optimized attention computation
- **Overlap Support**: Optional computation overlap
- **Memory Efficiency**: Reduces memory per GPU

### Best Practices

1. Use when:

   - Working with long sequences
   - Attention computation is memory-intensive
   - Need to process longer contexts

2. Configuration Tips:
   - Choose appropriate split dimension
   - Enable overlap for better performance
   - Consider communication overhead

## Pipeline Parallelism

Pipeline Parallelism splits model layers across GPUs, enabling efficient memory usage.

### Configuration

```python
from dream_trainer import DreamTrainerConfig
from dream_trainer.configs import DeviceParameters

config = DreamTrainerConfig(
    device_parameters=DeviceParameters(
        pipeline_parallel_size=2,  # Split across 2 GPUs
        pipeline_parallel_config={
            "num_microbatches": 4,  # Number of microbatches
            "schedule": "1F1B"  # Pipeline schedule
        }
    )
)

class MyTrainer(DreamTrainer):
    def configure_models(self):
        # Model is automatically split into pipeline stages
        self.model = self.model.to(self.device)
```

### Key Features

- **Layer Splitting**: Distribute model layers across GPUs
- **Microbatch Support**: Process multiple batches in pipeline
- **Efficient Scheduling**: Various pipeline schedules available
- **Memory Efficiency**: Each GPU holds only its layers

### Best Practices

1. Use when:

   - Model has many layers
   - Need to maximize GPU utilization
   - Memory efficiency is critical

2. Configuration Tips:
   - Choose appropriate number of microbatches
   - Select pipeline schedule based on model
   - Balance pipeline stages

## Combining Parallelism

Dream Trainer makes it easy to combine different types of parallelism.

### Example: FSDP + Tensor Parallel

```python
config = DreamTrainerConfig(
    device_parameters=DeviceParameters(
        data_parallel_size=2,  # 2-way data parallel
        tensor_parallel_size=2,  # 2-way tensor parallel
        fsdp_config={
            "sharding_strategy": "FULL_SHARD",
            "mixed_precision": True
        }
    )
)
```

### Example: Pipeline + Context Parallel

```python
config = DreamTrainerConfig(
    device_parameters=DeviceParameters(
        pipeline_parallel_size=2,  # 2-way pipeline parallel
        context_parallel_size=2,  # 2-way context parallel
        pipeline_parallel_config={
            "num_microbatches": 4,
            "schedule": "1F1B"
        }
    )
)
```

### Best Practices for Combining

1. **Start Simple**: Begin with one type of parallelism
2. **Add Gradually**: Add more parallelism as needed
3. **Monitor Performance**: Watch for communication overhead
4. **Balance Resources**: Ensure even distribution of work
5. **Consider Memory**: Account for memory requirements

## Common Issues

### Memory Issues

- **Out of Memory**: Reduce parallelism degree or enable mixed precision
- **Uneven Memory**: Balance pipeline stages or tensor sharding
- **Gradient Memory**: Use gradient checkpointing or FSDP

### Performance Issues

- **Slow Training**: Check communication overhead
- **Poor Scaling**: Verify batch size and parallelism configuration
- **Bottlenecks**: Profile to identify communication bottlenecks

### Debugging Tips

1. Start with small models and data
2. Enable detailed logging
3. Use PyTorch profiler
4. Monitor GPU utilization
5. Check communication patterns

## Next Steps

- Read the [Configuration Guide](configuration.md) for detailed settings
- Check [Examples](examples/advanced.md) for complete working code
- See [Best Practices](best-practices.md) for optimization tips
