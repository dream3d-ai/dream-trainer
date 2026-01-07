# Performance Tuning Guide

This guide covers performance optimization for Dream Trainer, from basic settings to advanced techniques. Drawing from torchtitan's performance patterns, we show how to maximize training throughput.

---

## Performance Checklist

Quick wins for better performance:

- [ ] Use `torch.bfloat16` for mixed precision
- [ ] Enable `torch.compile`
- [ ] Set optimal batch size
- [ ] Use FSDP prefetching
- [ ] Enable async checkpointing
- [ ] Set NCCL environment variables
- [ ] Use fast storage for data

---

## Basic Optimizations

### Mixed Precision Training

```python
import torch
from dream_trainer.configs import DeviceParameters


# BF16 (recommended for modern GPUs)
config = MyConfig(
    device_parameters=DeviceParameters(
        param_dtype=torch.bfloat16,   # Model weights
        reduce_dtype=torch.float32,    # Gradient reduction
    ),
)

# FP16 (older GPUs or specific requirements)
config = MyConfig(
    device_parameters=DeviceParameters(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        # Loss scaling handled automatically
    ),
)

# FP8 (H100 and newer)
from dream_trainer.callbacks import Fp8Quantization

config = MyConfig(
    device_parameters=DeviceParameters(
        param_dtype=torch.bfloat16,
    ),
    callbacks=CallbackCollection([
        Fp8Quantization(
            modules_to_quantize=["attention", "mlp"],
        ),
    ]),
)
```

### torch.compile

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        compile_model=True,         # Compile the model
        compiled_autograd=True,     # Compile backward pass too
    ),
)


# Custom compile options
class MyTrainer(DreamTrainer):
    def apply_compile(self):
        """Apply custom compilation."""
        self.model = torch.compile(
            self.model,
            mode="max-autotune",     # Maximize performance
            fullgraph=True,          # Full graph compilation
            dynamic=False,           # Static shapes
        )
```

### Optimal Batch Size

```python
def find_optimal_batch_size(trainer, start_batch_size=4, max_batch_size=512):
    """Find the largest batch size that fits in memory."""
    batch_size = start_batch_size

    while batch_size <= max_batch_size:
        try:
            # Create dummy batch
            trainer.config.training_parameters.train_batch_size = batch_size
            dummy_batch = trainer._create_dummy_batch(batch_size)

            # Run forward and backward
            output = trainer.training_step(dummy_batch, 0)
            trainer.optimizer.zero_grad()

            torch.cuda.empty_cache()
            print(f"Batch size {batch_size}: OK")

            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                optimal = batch_size // 2
                print(f"Optimal batch size: {optimal}")
                return optimal
            raise

    return batch_size // 2


# Use gradient accumulation for larger effective batch
config = MyConfig(
    training_parameters=TrainingParameters(
        train_batch_size=8,                  # Per-GPU micro-batch
        gradient_accumulation_steps=16,      # Effective batch = 8 * 16 * num_gpus
    ),
)
```

---

## FSDP Optimization

### FSDP Prefetching

```python
from dream_trainer.callbacks import OptimizeFSDP


config = MyConfig(
    device_parameters=DeviceParameters.FSDP(dp_shard=8),
    callbacks=CallbackCollection([
        OptimizeFSDP(
            prefetch=2,              # Prefetch 2 layers ahead
            trace_execution=True,    # Auto-detect optimal order
            backward_prefetch=True,  # Prefetch during backward too
        ),
    ]),
)
```

### Optimal Sharding Granularity

```python
class OptimallyShardedTrainer(DreamTrainer):
    def apply_fully_shard(self, fsdp_config):
        """Apply FSDP with optimal granularity."""
        from torch.distributed._composable.fsdp import fully_shard

        # Rule of thumb: Shard at ~100M-500M parameters
        param_threshold = 100_000_000

        for name, module in self.model.named_modules():
            # Skip small modules
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params < param_threshold:
                continue

            # Shard this module
            fully_shard(module, **fsdp_config)

        # Final wrap
        fully_shard(self.model, **fsdp_config)
```

### Rate-Limited All-Gather

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        dp_shard=8,
        # Limit concurrent all-gathers to reduce memory
        fsdp_limit_all_gathers=True,
    ),
)
```

---

## Communication Optimization

### NCCL Environment Variables

```bash
# Optimal NCCL settings for most clusters
export NCCL_IB_DISABLE=0           # Enable InfiniBand
export NCCL_IB_GID_INDEX=3         # GID index for RDMA
export NCCL_SOCKET_IFNAME=eth0     # Network interface
export NCCL_DEBUG=WARN             # Reduce debug overhead

# For NVLink systems
export NCCL_P2P_LEVEL=NVL          # Use NVLink for P2P
export NCCL_NET_GDR_LEVEL=2        # GPU Direct RDMA level

# For large clusters
export NCCL_NSOCKS_PERTHREAD=4     # Sockets per thread
export NCCL_SOCKET_NTHREADS=2      # Socket threads
export NCCL_BUFFSIZE=16777216      # 16MB buffer

# Tree reduction for large world sizes
export NCCL_ALGO=Tree              # Use tree algorithm
```

### Overlap Communication and Computation

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        dp_shard=8,
        # Enable async tensor parallel
        async_tensor_parallel=True,
        # Forward prefetching
        fsdp_forward_prefetch=True,
    ),
)


# Manual overlap with torch.cuda.Stream
class OverlappedTrainer(DreamTrainer):
    def training_step(self, batch, batch_idx):
        # Data transfer on separate stream
        data_stream = torch.cuda.Stream()
        with torch.cuda.stream(data_stream):
            x = batch['input'].to(self.device, non_blocking=True)
            y = batch['target'].to(self.device, non_blocking=True)

        # Compute on default stream
        data_stream.synchronize()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        self.backward(loss)

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)

        return {"loss": loss}
```

---

## Memory Optimization

### Activation Checkpointing

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        checkpoint_activations=True,
    ),
)


# Custom checkpointing policy
class SelectiveCheckpointingTrainer(DreamTrainer):
    def apply_activation_checkpointing(self):
        """Apply selective checkpointing."""
        from torch.utils.checkpoint import checkpoint_sequential

        # Only checkpoint transformer blocks
        for layer in self.model.layers:
            # Checkpoint attention (memory-intensive)
            layer.attention = partial(
                checkpoint,
                layer.attention,
                use_reentrant=False,
            )
```

### CPU Offloading

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        dp_shard=8,
        cpu_offload=True,  # Offload optimizer states to CPU
    ),
)
```

### Memory-Efficient Attention

```python
# Use Flash Attention (automatic with PyTorch 2.0+)
class EfficientAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        # PyTorch's scaled_dot_product_attention uses Flash Attention when available
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0 if not self.training else self.dropout,
            is_causal=True,  # For decoder models
        )
```

---

## Data Loading Optimization

### Efficient DataLoader

```python
def create_optimized_dataloader(dataset, batch_size, num_workers=None):
    """Create an optimized DataLoader."""
    import multiprocessing

    if num_workers is None:
        num_workers = min(8, multiprocessing.cpu_count())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,           # Faster GPU transfer
        persistent_workers=True,    # Keep workers alive
        prefetch_factor=4,          # Prefetch batches
        drop_last=True,             # Consistent batch sizes
    )


# Use async data loading
class AsyncDataLoader:
    """DataLoader that prefetches to GPU asynchronously."""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        for batch in self.dataloader:
            # Transfer on separate stream
            with torch.cuda.stream(self.stream):
                batch = self._to_device(batch)
            yield batch

    def _to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(v.to(self.device, non_blocking=True) for v in batch)
        return batch.to(self.device, non_blocking=True)
```

### Dataset Preprocessing

```python
# Preprocess data offline
def preprocess_dataset(raw_path, output_path, tokenizer):
    """Preprocess and save tokenized data."""
    import numpy as np

    # Tokenize all data
    tokens = []
    with open(raw_path) as f:
        for line in tqdm(f):
            tokens.extend(tokenizer.encode(line))

    # Save as memory-mapped array
    tokens = np.array(tokens, dtype=np.int32)
    np.save(output_path, tokens)


# Use memory-mapped dataset
class MemmapDataset(Dataset):
    def __init__(self, path, seq_length):
        self.data = np.load(path, mmap_mode='r')
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return torch.from_numpy(
            self.data[idx:idx + self.seq_length].copy()
        ).long()
```

---

## Profiling and Benchmarking

### Built-in Profiling

```python
from dream_trainer.callbacks import ProfileCallback


config = MyConfig(
    callbacks=CallbackCollection([
        ProfileCallback(
            profile_steps=range(10, 20),
            output_dir="./profiles",
            activities=["cpu", "cuda"],
            schedule=torch.profiler.schedule(
                wait=1, warmup=2, active=6, repeat=1
            ),
        ),
    ]),
)
```

### Custom Benchmarking

```python
def benchmark_throughput(trainer, num_steps=100, warmup_steps=10):
    """Benchmark training throughput."""
    import time

    # Warmup
    for i, batch in enumerate(trainer.train_dataloader):
        if i >= warmup_steps:
            break
        trainer.training_step(batch, i)

    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    tokens_processed = 0

    for i, batch in enumerate(trainer.train_dataloader):
        if i >= num_steps:
            break

        trainer.training_step(batch, i)
        tokens_processed += batch['input_ids'].numel()

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # Report
    samples_per_sec = num_steps * trainer.config.training_parameters.train_batch_size / elapsed
    tokens_per_sec = tokens_processed / elapsed

    print(f"Throughput: {samples_per_sec:.1f} samples/sec")
    print(f"Throughput: {tokens_per_sec/1000:.1f}K tokens/sec")
    print(f"Time per step: {elapsed/num_steps*1000:.1f} ms")

    return {
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_step": elapsed / num_steps * 1000,
    }
```

### Model FLOPs Profiling

```python
def estimate_model_flops(model, input_shape):
    """Estimate model FLOPs."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode() as flop_counter:
        dummy_input = torch.randn(*input_shape).cuda()
        model(dummy_input)

    total_flops = flop_counter.get_total_flops()
    print(f"Model FLOPs: {total_flops/1e12:.2f} TFLOPs")
    return total_flops
```

---

## Performance Targets

### Expected Throughput by Hardware

| GPU | Batch Size | BF16 Throughput | Notes |
|-----|------------|-----------------|-------|
| A100 40GB | 4-8 | 30-50K tokens/sec | Per GPU |
| A100 80GB | 8-16 | 40-60K tokens/sec | Per GPU |
| H100 | 8-16 | 80-120K tokens/sec | Per GPU |
| 8x A100 | 32-64 | 200-400K tokens/sec | FSDP |
| 8x H100 | 64-128 | 400-800K tokens/sec | FSDP |

### Optimization Impact

| Optimization | Typical Speedup | Memory Reduction |
|--------------|-----------------|------------------|
| BF16 | 1.5-2x | 50% |
| torch.compile | 1.2-1.5x | - |
| FSDP prefetch | 1.1-1.3x | - |
| Flash Attention | 1.3-2x | 40-60% |
| Activation checkpointing | 0.8x | 30-50% |
| FP8 (H100) | 1.5-2x | 50% |

---

## Performance Configuration Templates

### Single GPU (Maximum Throughput)

```python
config = MyConfig(
    device_parameters=DeviceParameters.SINGLE_DEVICE(
        param_dtype=torch.bfloat16,
        compile_model=True,
    ),
    training_parameters=TrainingParameters(
        train_batch_size=32,  # Maximize batch size
    ),
)
```

### Multi-GPU (8x A100)

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        dp_shard=8,
        param_dtype=torch.bfloat16,
        compile_model=True,
        compiled_autograd=True,
    ),
    training_parameters=TrainingParameters(
        train_batch_size=8,
        gradient_accumulation_steps=4,
    ),
    callbacks=CallbackCollection([
        OptimizeFSDP(prefetch=2),
    ]),
)
```

### Large Model (Memory-Constrained)

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        dp_shard=8,
        tensor_parallel=2,
        param_dtype=torch.bfloat16,
        checkpoint_activations=True,
    ),
    training_parameters=TrainingParameters(
        train_batch_size=2,
        gradient_accumulation_steps=16,
    ),
)
```

### Multi-Node Cluster

```python
config = MyConfig(
    device_parameters=DeviceParameters(
        dp_shard=64,           # 8 nodes * 8 GPUs
        tensor_parallel=8,     # TP within node
        dp_replicate=1,        # No replication
        param_dtype=torch.bfloat16,
        compile_model=True,
        async_tensor_parallel=True,
    ),
    training_parameters=TrainingParameters(
        train_batch_size=4,
        gradient_accumulation_steps=8,
    ),
)
```

---

## Next Steps

- [Debugging Guide](debugging.md): Debug performance issues
- [Parallelism Guide](parallelism.md): Optimize distributed training
- [Callbacks](callbacks.md): Performance-related callbacks
