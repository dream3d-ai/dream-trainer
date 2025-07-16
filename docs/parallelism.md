# Parallelism Guide

Dream Trainer provides first-class support for all modern parallelism strategies in PyTorch. This guide covers everything from basic data parallelism to advanced self-parallelizing models using `fsdp2_utils`.

## Table of Contents

- [Overview](#overview)
- [Understanding Memory and Batch Size Constraints](#understanding-memory-and-batch-size-constraints)
- [Parallelism Strategies](#parallelism-strategies)
- [Network Topology Considerations](#network-topology-considerations)
- [Basic Configuration](#basic-configuration)
- [Self-Parallelizing Models with fsdp2_utils](#self-parallelizing-models-with-fsdp2_utils)
- [Manual Parallelism Implementation](#manual-parallelism-implementation)
- [Combining Parallelism Strategies](#combining-parallelism-strategies)
- [Performance Optimization](#performance-optimization)
- [Debugging Tips](#debugging-tips)

## Overview

Dream Trainer supports five types of parallelism, all built on PyTorch's DTensor infrastructure:

1. **Data Parallelism (DP)**: Split batches across devices
   - `dp_replicate`: Traditional DDP replication
   - `dp_shard`: FSDP2 sharding

2. **Tensor Parallelism (TP)**: Split model layers across devices

3. **Pipeline Parallelism (PP)**: Split model stages across devices

4. **Context Parallelism (CP)**: Split sequence dimension for long contexts

5. **Hybrid Strategies**: Combine multiple types (e.g., HSDP = DDP + FSDP)

## Understanding Memory and Batch Size Constraints

Before diving into parallelism strategies, it's crucial to understand what consumes memory during training and how batch size affects training dynamics.

### Memory Breakdown During Training

According to [Jeremy Jordan's analysis](https://www.jeremyjordan.me/distributed-training/), training a model requires keeping several components in memory:

1. **Model Parameters**: The learnable weights (e.g., 405B parameters = ~810GB in FP16)
2. **Optimizer States**: 
   - SGD: Just parameters (1x memory)
   - Adam/AdamW: Parameters + first moment + second moment (3x memory)
3. **Model Activations**: Intermediate values needed for backpropagation
   - Scales with batch size and model architecture
   - Can be the dominant memory consumer for large batches
4. **Gradients**: Same size as model parameters
5. **Input Data**: The actual batch being processed

```python
# Example memory calculation for a 7B parameter model
model_params = 7e9 * 2  # 14GB in FP16
optimizer_states = model_params * 3  # 42GB for AdamW
gradients = model_params  # 14GB
# Total: 70GB + activations + data
```

### The Science of Batch Size Scaling

Not all batch sizes are created equal. As explained in ["An Empirical Model of Large-Batch Training"](https://www.jeremyjordan.me/distributed-training/), there are two distinct regimes:

1. **Perfect Scaling Regime**: When batch size is small, you can double the batch size and double the learning rate to train in half the steps
2. **Ineffective Scaling Regime**: Beyond a critical batch size, increasing batch size provides diminishing returns

The transition point (called the **gradient noise scale**) depends on your data and model. Importantly, this transition point *increases* during training, which is why models like Llama 3.1 progressively increase batch size:

```python
# Llama 3.1 training schedule
initial_batch = 4M tokens
after 252M tokens: batch_size = 8M
after 2.87T tokens: batch_size = 16M
```

### Memory Reduction Techniques

When you hit memory limits on a single GPU, you have several options before resorting to parallelism:

```python
# 1. Gradient Accumulation (trade compute for memory)
config = TrainingParameters(
    train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch = 32
)

# 2. Activation Checkpointing (recompute vs store)
config = DeviceParameters(
    checkpoint_activations=True,  # ~30% memory savings
)

# 3. Mixed Precision (reduce precision)
config = DeviceParameters(
    param_dtype=torch.bfloat16,  # 2x memory savings vs FP32
)
```

## Parallelism Strategies

### Understanding the Trade-offs

Each parallelism strategy makes different trade-offs between memory savings, communication overhead, and implementation complexity:

| Strategy | Memory Savings | Communication | Best For |
|----------|---------------|---------------|----------|
| DDP | None | All-reduce gradients (once per step) | Small models, high bandwidth |
| FSDP2 | High | All-gather params + reduce-scatter grads | Large models |
| TP | Medium | All-reduce activations (multiple per layer) | Wide models (large hidden dims) |
| PP | High | Point-to-point activations | Deep models (many layers) |
| CP | Medium | All-to-all for attention | Long sequences |

### Data Parallelism: The Foundation

Data parallelism is the simplest and most common form of distributed training. Each GPU maintains a complete copy of the model and processes different batches:

```python
# Traditional DDP - each GPU has full model
config = DeviceParameters(dp_replicate=8)  # 8 GPUs

# FSDP2 - shards model across GPUs
config = DeviceParameters(dp_shard=8)  # 8 GPUs, ~8x memory reduction
```

**When to use DDP:**
- Model fits on single GPU
- High-bandwidth interconnect available
- Want simplest implementation

**When to use FSDP2:**
- Model too large for single GPU
- Need to train larger models with same hardware
- Can tolerate some communication overhead

### Tensor Parallelism: Splitting Layers

Tensor parallelism splits individual layers across GPUs. As [explained by Jeremy Jordan](https://www.jeremyjordan.me/distributed-training/), there are two main approaches:

#### Column Partitioning
Splits weight matrix along output dimension:

```python
# Weight W: [input_dim, output_dim]
# Each GPU gets W_i: [input_dim, output_dim/n_gpus]

# Forward: X @ W_i → Y_i (then all-gather)
# Backward: Gradient flows naturally
```

#### Row Partitioning  
Splits weight matrix along input dimension:

```python
# Weight W: [input_dim, output_dim]
# Each GPU gets W_i: [input_dim/n_gpus, output_dim]

# Forward: X_i @ W_i → partial Y (then all-reduce)
# Backward: Need all-gather for input gradients
```

#### Optimizing Communication

The Megatron-LM paper showed how clever partitioning reduces communication:

```python
class OptimizedMLP(nn.Module, ParallelPlan):
    def parallelize_plan(self):
        return {
            # First layer: column partition (output split)
            "fc1": colwise_parallel(self.fc1),
            
            # Activation: computed locally on each GPU
            
            # Second layer: row partition (input split)
            # Takes split input directly - no communication!
            "fc2": rowwise_parallel(self.fc2),
            
            # Only one all-reduce at the end
        }
```

This pattern reduces communication by 50% compared to naive partitioning!

### Pipeline Parallelism: Splitting Stages

Pipeline parallelism splits the model into sequential stages, with each GPU responsible for a subset of layers:

```python
# Model split across 4 GPUs
GPU 0: Embedding + Layers[0:8]
GPU 1: Layers[8:16]  
GPU 2: Layers[16:24]
GPU 3: Layers[24:32] + Output

# Microbatching keeps GPUs busy
microbatch_size = batch_size // num_microbatches
```

**Advantages:**
- Each GPU only stores its layers (high memory savings)
- Only forward/backward activations communicated
- Works well across nodes with slower interconnect

**Disadvantages:**
- Pipeline bubbles reduce efficiency
- Requires careful load balancing
- More complex implementation

### Context Parallelism: Splitting Sequences

For extremely long sequences, context parallelism splits the sequence dimension:

```python
# Sequence length 128K split across 4 GPUs
GPU 0: tokens[0:32K]
GPU 1: tokens[32K:64K]
GPU 2: tokens[64K:96K]  
GPU 3: tokens[96K:128K]

# Attention requires all-to-all communication
```

## Network Topology Considerations

Modern GPU clusters have hierarchical network structures that significantly impact parallelism choices. As noted in the [Llama 3.1 training details](https://www.jeremyjordan.me/distributed-training/):

```
Within node: 8 GPUs connected via NVLink (900 GB/s)
Within rack: 2 nodes connected via network switch
Within pod: 192 racks (3,072 GPUs)
Full cluster: 8 pods (24,576 GPUs)
```

### Optimal Parallelism Placement

Based on network topology, place parallelism strategies hierarchically:

```python
# Optimal configuration for hierarchical networks
config = DeviceParameters(
    # Within node (NVLink - highest bandwidth)
    tensor_parallel=8,  # Requires frequent communication
    
    # Across nodes within rack (InfiniBand - medium bandwidth)  
    pipeline_parallel=2,  # Only activations communicated
    
    # Across racks (Ethernet - lowest bandwidth)
    dp_shard=16,  # Only gradient sync required
)

# Total: 8 * 2 * 16 = 256 GPUs
```

### Example: Llama 3.1 405B Configuration

The Llama 3.1 training used a carefully optimized setup:

```python
# Llama 3.1 405B parallelism
tensor_parallel = 8      # Within node (NVLink)
pipeline_parallel = 16   # Across nodes
data_parallel = 128      # Across everything

# Total: 16,384 GPUs
# Memory per GPU: ~50GB (405B params / 8 TP / 16 PP)
```

## Basic Configuration

Configure parallelism through `DeviceParameters`:

```python
from dream_trainer.configs import DeviceParameters

config = DeviceParameters(
    # Data parallelism
    dp_replicate=2,      # 2-way DDP
    dp_shard=4,          # 4-way FSDP within each DDP group
    
    # Model parallelism
    tensor_parallel=2,   # 2-way TP
    pipeline_parallel=4, # 4 pipeline stages
    context_parallel=2,  # 2-way sequence splitting
    
    # Optimizations
    compile_model=True,
    loss_parallel=True,  # Parallel loss with TP
    async_tensor_parallel=True,  # Overlap TP communication
)
```

## Self-Parallelizing Models with fsdp2_utils

The most elegant approach to parallelism in Dream Trainer is using `fsdp2_utils` to create models that know how to parallelize themselves. This encapsulates complex sharding logic within the model definition.

### The Pattern: Models That Parallelize Themselves

Instead of implementing parallelism in the trainer, models inherit from `FullyShard` and `ParallelPlan`:

```python
from typing import Any
import torch.nn as nn
from fsdp2_utils import FullyShard, ParallelPlan, apply_tensor_parallel, apply_fully_shard
from fsdp2_utils.tensor_parallel.plan import (
    colwise_parallel, 
    rowwise_parallel,
    sequence_parallel,
    prepare_module_input
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.placement_types import Replicate, Shard

class TransformerModel(nn.Module, FullyShard, ParallelPlan):
    """A transformer that knows how to parallelize itself"""
    
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def fully_shard(self, config: dict[str, Any]):
        """Define FSDP sharding strategy"""
        # Shard each transformer layer independently
        for layer in self.layers:
            fully_shard(layer, **config)
        
        # Shard embeddings and output separately
        fully_shard(self.embeddings, **config)
        fully_shard(self.lm_head, **config)
    
    def parallelize_plan(self):
        """Define tensor parallel strategy"""
        plan = {
            # Embeddings: row-wise parallel (vocab dimension)
            "embeddings": rowwise_parallel(
                self.embeddings,
                input_layouts=Replicate(),  # Input tokens are replicated
                output_layouts=Shard(1),    # Output is sharded on sequence dim
            ),
            
            # Final layer norm: sequence parallel
            "norm": sequence_parallel(self.norm),
            
            # LM head: column-wise parallel (vocab dimension)
            "lm_head": colwise_parallel(
                self.lm_head,
                input_layouts=Shard(1),     # Input is sharded
                output_layouts=Replicate(), # Output is replicated
                use_local_output=True,
            ),
        }
        
        # Add plans for each transformer layer
        for i, layer in enumerate(self.layers):
            plan.update(layer.parallelize_plan(prefix=f"layers.{i}"))
        
        return plan
```

### Implementing Parallelism in Submodules

Each component defines its own parallelism strategy:

```python
class TransformerLayer(nn.Module, ParallelPlan):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
    
    def parallelize_plan(self, prefix=""):
        """Define TP plan for this layer"""
        def add_prefix(name):
            return f"{prefix}.{name}" if prefix else name
        
        return {
            # Attention: QKV column-wise, O row-wise
            add_prefix("attention.q_proj"): colwise_parallel(self.attention.q_proj),
            add_prefix("attention.k_proj"): colwise_parallel(self.attention.k_proj),
            add_prefix("attention.v_proj"): colwise_parallel(self.attention.v_proj),
            add_prefix("attention.o_proj"): rowwise_parallel(
                self.attention.o_proj,
                input_layouts=Shard(-1),  # Sharded on head dimension
            ),
            
            # MLP: standard Megatron-style parallelism
            add_prefix("mlp.gate_proj"): colwise_parallel(self.mlp.gate_proj),
            add_prefix("mlp.up_proj"): colwise_parallel(self.mlp.up_proj),
            add_prefix("mlp.down_proj"): rowwise_parallel(
                self.mlp.down_proj,
                input_layouts=Shard(-1),  # Sharded on hidden dimension
            ),
            
            # Layer norms use sequence parallelism
            add_prefix("norm1"): sequence_parallel(self.norm1),
            add_prefix("norm2"): sequence_parallel(self.norm2),
        }
```

### Simplified Trainer Implementation

With self-parallelizing models, trainers become remarkably simple:

```python
from fsdp2_utils import apply_tensor_parallel, apply_fully_shard

class MyTrainer(DreamTrainer):
    def configure_models(self):
        # Model knows how to parallelize itself!
        self.model = TransformerModel(self.config)
    
    def apply_tensor_parallel(self, tp_mesh: DeviceMesh):
        """Just delegate to the model's built-in TP strategy"""
        apply_tensor_parallel(self.model, tp_mesh)
    
    def apply_fully_shard(self, config: dict[str, Any]):
        """Just delegate to the model's built-in FSDP strategy"""
        apply_fully_shard(self.model, config)
```

### Advanced: Heterogeneous Parallelism for Multi-Modal Models

Different model components can use different strategies:

```python
class VisionLanguageModel(nn.Module, FullyShard, ParallelPlan):
    """Multi-modal model with component-specific parallelism"""
    
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = VisionTransformer(config.vision)
        self.text_encoder = TextTransformer(config.text)
        self.cross_attention = CrossModalAttention(config)
        self.projection = nn.Linear(config.hidden_size, config.output_size)
    
    def fully_shard(self, config: dict[str, Any]):
        """Different sharding for different components"""
        # Vision: larger sharding units (less communication)
        vision_config = {**config, "min_num_params_per_shard": 50_000_000}
        self.vision_encoder.fully_shard(vision_config)
        
        # Text: standard sharding
        self.text_encoder.fully_shard(config)
        
        # Cross-attention: aggressive sharding (memory intensive)
        cross_config = {**config, "min_num_params_per_shard": 10_000_000}
        fully_shard(self.cross_attention, **cross_config)
    
    def parallelize_plan(self):
        """Different TP strategies for different modalities"""
        return {
            # Vision doesn't use TP (compute bound, not memory bound)
            "vision_encoder": prepare_module_input(
                self.vision_encoder,
                input_layouts=Replicate(),
            ),
            
            # Text uses full TP
            **{f"text_encoder.{k}": v 
               for k, v in self.text_encoder.parallelize_plan().items()},
            
            # Cross-attention uses custom strategy
            "cross_attention.q_proj": colwise_parallel(
                self.cross_attention.q_proj,
                input_layouts=Replicate(),  # From vision
            ),
            "cross_attention.kv_proj": colwise_parallel(
                self.cross_attention.kv_proj,
                input_layouts=Shard(1),     # From text
            ),
        }
```

### Benefits of Self-Parallelizing Models

1. **Encapsulation**: Parallelism logic lives with model definition
2. **Reusability**: Same model works in any trainer
3. **Clarity**: Structure and strategy are co-located
4. **Composability**: Complex models built from simple parallel components
5. **Type Safety**: IDE understands the parallelism interface

## Manual Parallelism Implementation

For cases where you need fine-grained control:

### Manual Tensor Parallelism

```python
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

class MyTrainer(DreamTrainer):
    def apply_tensor_parallel(self, tp_mesh: DeviceMesh):
        # Manual TP for specific layers
        for layer in self.model.layers:
            plan = {
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(),
                "mlp.w1": ColwiseParallel(),
                "mlp.w2": RowwiseParallel(),
            }
            parallelize_module(layer, tp_mesh, plan)
```

### Manual FSDP2

```python
from torch.distributed._composable.fsdp import fully_shard

class MyTrainer(DreamTrainer):
    def apply_fully_shard(self, config: dict[str, Any]):
        # Custom sharding per layer
        for i, layer in enumerate(self.model.layers):
            if i < 10:  # First 10 layers
                fully_shard(layer, **config)
            else:  # Later layers use different config
                custom_config = {**config, "reshard_after_forward": False}
                fully_shard(layer, **custom_config)
```

### Manual Pipeline Parallelism

```python
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage

class MyTrainer(DreamTrainer):
    def apply_pipeline_parallel(self, pp_mesh: DeviceMesh):
        # Define split points
        mb = [self.model.embeddings, *self.model.layers, self.model.output]
        
        # Create pipeline stages
        stages = [
            PipelineStage(mb[:8], 0),   # First 8 layers
            PipelineStage(mb[8:16], 1),  # Next 8 layers
            PipelineStage(mb[16:24], 2), # Next 8 layers  
            PipelineStage(mb[24:], 3),   # Rest
        ]
        
        # Create schedule
        schedule = pipeline(
            stages,
            pp_mesh,
            mb_size=self.config.pipeline_parallel_microbatch_size,
        )
        
        return {"model": (schedule, stages, True, True)}
```

## Combining Parallelism Strategies

### Real-World Examples

Based on model size and available hardware, here are recommended configurations:

#### Small Model (< 1B parameters)
```python
# Fits on single GPU - use pure data parallelism
config = DeviceParameters(
    dp_replicate=num_gpus,  # Simple DDP
    compile_model=True,     # Optimize single GPU perf
)
```

#### Medium Model (1B - 70B parameters)
```python
# Needs FSDP for memory, optional TP for speed
config = DeviceParameters(
    dp_shard=8,          # FSDP for memory efficiency
    tensor_parallel=2,   # Optional: faster but more GPUs
)
```

#### Large Model (70B - 500B parameters)
```python
# Requires both memory and compute parallelism
config = DeviceParameters(
    tensor_parallel=8,    # Split within nodes
    pipeline_parallel=4,  # Split across nodes
    dp_shard=16,         # FSDP across everything
)
```

#### Extreme Scale (500B+ parameters)
```python
# Everything at maximum scale
config = DeviceParameters(
    tensor_parallel=8,     # Max TP within node
    pipeline_parallel=16,  # Many pipeline stages
    dp_replicate=2,       # HSDP: DDP across pods
    dp_shard=64,          # FSDP within pods
    context_parallel=4,    # For long sequences
)
```

### 3D Parallelism (DP + TP + PP)

```python
config = DeviceParameters(
    dp_shard=4,          # 4-way FSDP
    tensor_parallel=2,   # 2-way TP  
    pipeline_parallel=2, # 2 pipeline stages
    # Total: 4 * 2 * 2 = 16 GPUs
)
```

### HSDP (Hybrid Sharded Data Parallel)

HSDP combines the benefits of DDP and FSDP by creating a hierarchy:

```python
config = DeviceParameters(
    dp_replicate=2,  # 2 DDP groups (across pods)
    dp_shard=4,      # 4-way FSDP within each group
)

# Communication pattern:
# - Gradient reduce-scatter within FSDP groups (high bandwidth)
# - Gradient all-reduce across DDP groups (low bandwidth)
# Result: Less cross-pod communication
```

### Context Parallel + Tensor Parallel

```python
config = DeviceParameters(
    context_parallel=4,  # 4-way sequence splitting
    tensor_parallel=2,   # 2-way tensor parallel
    # Enables 4x longer sequences with 2x model parallel
)
```

## Performance Optimization

### Optimizing for Your Hardware

1. **Measure Your Baseline**
```python
# Profile single GPU performance first
with torch.profiler.profile() as prof:
    trainer.fit()
    
# Key metrics:
# - GPU utilization (target: >90%)
# - Memory bandwidth utilization
# - Time spent in communication vs compute
```

2. **Scale Gradually**
```python
# Start simple, add parallelism incrementally
configs = [
    DeviceParameters(dp_replicate=8),  # Pure DDP
    DeviceParameters(dp_shard=8),      # Pure FSDP
    DeviceParameters(dp_shard=4, tensor_parallel=2),  # FSDP+TP
    DeviceParameters(dp_shard=2, tensor_parallel=2, pipeline_parallel=2),  # 3D
]
```

3. **Monitor Scaling Efficiency**
```python
# Perfect scaling: 2x GPUs = 2x throughput
scaling_efficiency = (
    throughput_ngpus / throughput_1gpu
) / n_gpus

# Good: >0.8
# Okay: 0.6-0.8  
# Poor: <0.6 (reconsider strategy)
```

### 1. FSDP Prefetching

```python
from dream_trainer.callbacks import OptimizeFSDP

config.callbacks = CallbackCollection([
    OptimizeFSDP(prefetch=2),  # Prefetch next 2 modules
])
```

### 2. Async Tensor Parallelism

```python
config = DeviceParameters(
    tensor_parallel=4,
    async_tensor_parallel=True,  # Overlap TP comms
)
```

### 3. Compiled Autograd

```python
config = DeviceParameters(
    compile_model=True,
    enable_compiled_autograd=True,  # Compile backward too
)
```

### 4. Communication Optimization

```python
# Set NCCL environment variables
import os
os.environ["NCCL_ALGO"] = "Tree"  # Better for small messages
os.environ["NCCL_PROTO"] = "Simple"  # Lower latency
os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
os.environ["NCCL_SOCKET_NTHREADS"] = "8"
```

### 5. Batch Size Optimization

Based on gradient noise scale theory:

```python
# Start with smaller batch size
initial_batch_size = 256

# Measure gradient noise scale
# (See "An Empirical Model of Large-Batch Training")

# Progressively increase batch size during training
schedule = {
    0: 256,        # Initial: high gradient noise
    100_000: 512,  # Model improving, can use larger batches
    500_000: 1024, # Near convergence, maximize efficiency
}
```

## Debugging Tips

### 1. Verify Sharding

```python
def training_step(self, batch, batch_idx):
    if batch_idx == 0:
        # Check parameter sharding
        for name, param in self.model.named_parameters():
            if hasattr(param, 'placements'):
                print(f"{name}: {param.placements}")
```

### 2. Monitor Memory Usage

```python
from dream_trainer.utils import log_memory_usage

def training_step(self, batch, batch_idx):
    log_memory_usage("before_forward")
    output = self.model(batch)
    log_memory_usage("after_forward")
```

### 3. Profile Communication

```python
import torch.distributed as dist

# Enable NCCL debug logging
os.environ["NCCL_DEBUG"] = "INFO"

# Profile with PyTorch profiler
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    trainer.fit()
```

### 4. Test Incrementally

1. Start with single GPU
2. Add DP (either DDP or FSDP)
3. Add TP
4. Add PP
5. Combine strategies

## Best Practices

1. **Understand Your Bottleneck**
   - Memory bound? → Use FSDP or PP
   - Compute bound? → Use TP for large matmuls
   - Communication bound? → Optimize placement based on topology

2. **Profile Before Optimizing**
   - Measure baseline single-GPU performance
   - Identify whether you're compute or memory bound
   - Profile communication patterns

3. **Consider Total System Design**
   - Network topology (NVLink vs InfiniBand vs Ethernet)
   - Model architecture (depth vs width)
   - Batch size requirements (gradient noise scale)

4. **Use Self-Parallelizing Models**
   - Encapsulate complexity in model definitions
   - Makes experimentation easier
   - Improves code reusability

5. **Test Scaling Incrementally**
   - Start with single GPU
   - Add data parallelism
   - Add model parallelism only if needed
   - Verify near-linear scaling at each step

## Next Steps

- Read the [Configuration Guide](configuration.md) for detailed parameter options
- Check [Examples](examples/advanced.md) for real-world parallelism usage
- See [Performance Guide](performance.md) for optimization tips
- Review [Jeremy Jordan's distributed training guide](https://www.jeremyjordan.me/distributed-training/) for more theory

---

**Remember**: The best parallelism strategy depends on your model architecture, hardware topology, and training requirements. Dream Trainer gives you the flexibility to experiment and find what works best! 🚀
