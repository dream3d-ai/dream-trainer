# Debugging Guide

This guide covers debugging techniques for Dream Trainer, from simple logging to advanced distributed debugging. Drawing from torchtitan's debugging patterns, we cover common issues and their solutions.

---

## Quick Debugging Checklist

Before diving deep, check these common issues:

- [ ] Is CUDA available? (`torch.cuda.is_available()`)
- [ ] Are all GPUs visible? (`torch.cuda.device_count()`)
- [ ] Is the correct PyTorch version installed? (`torch.__version__`)
- [ ] Are environment variables set correctly?
- [ ] Does the model fit in memory?
- [ ] Are all ranks reaching the same point?

---

## Debug Mode Configuration

### Enable Comprehensive Logging

```python
from dataclasses import dataclass
from dream_trainer import DreamTrainerConfig


@dataclass
class DebugConfig(DreamTrainerConfig):
    """Configuration with debugging enabled."""
    debug_mode: bool = True
    log_level: str = "DEBUG"
    profile_first_n_steps: int = 10
    trace_memory: bool = True
    validate_tensors: bool = True


def enable_debug_environment():
    """Set environment variables for debugging."""
    import os

    # PyTorch debugging
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    # NCCL debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

    # CUDA debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA calls

    # Python debugging
    import faulthandler
    faulthandler.enable()
```

### Debug Callback

```python
from dream_trainer.callbacks import Callback
import torch


class DebugCallback(Callback):
    """Comprehensive debugging callback."""

    def __init__(
        self,
        check_nan: bool = True,
        check_inf: bool = True,
        log_shapes: bool = False,
        log_memory: bool = True,
        validate_grads: bool = True,
    ):
        super().__init__()
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.log_shapes = log_shapes
        self.log_memory = log_memory
        self.validate_grads = validate_grads

    def post_train_step(self, output, batch_idx):
        # Check for NaN/Inf in loss
        if self.check_nan or self.check_inf:
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    if self.check_nan and torch.isnan(value).any():
                        raise RuntimeError(f"NaN detected in {key} at step {batch_idx}")
                    if self.check_inf and torch.isinf(value).any():
                        raise RuntimeError(f"Inf detected in {key} at step {batch_idx}")

        # Log memory usage
        if self.log_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[Step {batch_idx}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def pre_optimizer_step(self, optimizer_name):
        if not self.validate_grads:
            return

        model = self.trainer.get_model_by_optimizer(optimizer_name)
        if model is None:
            return

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            if torch.isnan(param.grad).any():
                raise RuntimeError(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                raise RuntimeError(f"Inf gradient in {name}")

            grad_norm = param.grad.norm().item()
            if grad_norm > 1000:
                print(f"Warning: Large gradient norm {grad_norm:.2f} in {name}")
```

---

## Distributed Debugging

### Debugging Multi-GPU Issues

```python
import torch
import torch.distributed as dist


def debug_distributed_state():
    """Print distributed training state on all ranks."""
    if not dist.is_initialized():
        print("Distributed not initialized")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}")
    print(f"[Rank {rank}] Device: {torch.cuda.current_device()}")
    print(f"[Rank {rank}] Device name: {torch.cuda.get_device_name()}")

    # Check NCCL
    if torch.cuda.is_available():
        tensor = torch.ones(1).cuda()
        dist.all_reduce(tensor)
        print(f"[Rank {rank}] All-reduce test: {tensor.item()}")


def verify_model_sync():
    """Verify all ranks have the same model parameters."""
    model = trainer.model

    for name, param in model.named_parameters():
        # Get parameter hash
        param_sum = param.data.sum().item()

        # Gather from all ranks
        all_sums = [None] * dist.get_world_size()
        dist.all_gather_object(all_sums, param_sum)

        if dist.get_rank() == 0:
            if len(set(all_sums)) > 1:
                print(f"Parameter mismatch in {name}: {all_sums}")
```

### Debugging FSDP

```python
from torch.distributed._composable.fsdp import FSDPModule


def debug_fsdp_state(model):
    """Debug FSDP sharding state."""
    for name, module in model.named_modules():
        if isinstance(module, FSDPModule):
            print(f"FSDP Module: {name}")
            print(f"  Params sharded: {module.params_sharded}")
            print(f"  Grads sharded: {module.grads_sharded}")

            # Check parameter distribution
            for pname, param in module.named_parameters(recurse=False):
                if hasattr(param, 'placements'):
                    print(f"  {pname}: {param.placements}")


def debug_dtensor_sharding(tensor, name="tensor"):
    """Debug DTensor sharding layout."""
    if not hasattr(tensor, 'placements'):
        print(f"{name} is not a DTensor")
        return

    print(f"{name}:")
    print(f"  Global shape: {tensor.shape}")
    print(f"  Local shape: {tensor._local_tensor.shape}")
    print(f"  Placements: {tensor.placements}")
    print(f"  Device mesh: {tensor.device_mesh}")
```

### Debugging Tensor Parallelism

```python
def debug_tensor_parallel(trainer):
    """Debug tensor parallel sharding."""
    if trainer.world.tp_mesh is None:
        print("Tensor parallelism not enabled")
        return

    print(f"TP Mesh: {trainer.world.tp_mesh}")
    print(f"TP Degree: {trainer.world.tp_mesh.size()}")

    for name, param in trainer.model.named_parameters():
        if hasattr(param, 'placements'):
            placements = param.placements
            local_shape = param._local_tensor.shape
            global_shape = param.shape

            if any(p.is_shard() for p in placements):
                print(f"{name}:")
                print(f"  Global: {global_shape} -> Local: {local_shape}")
                print(f"  Placements: {placements}")
```

---

## Memory Debugging

### Track Memory Usage

```python
import torch


class MemoryTracker:
    """Track GPU memory usage during training."""

    def __init__(self):
        self.snapshots = []

    def snapshot(self, label: str):
        """Take a memory snapshot."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.snapshots.append({
                'label': label,
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
            })

    def report(self):
        """Print memory report."""
        print("\n=== Memory Report ===")
        for snap in self.snapshots:
            print(f"{snap['label']}:")
            print(f"  Allocated: {snap['allocated'] / 1e9:.2f} GB")
            print(f"  Reserved: {snap['reserved'] / 1e9:.2f} GB")
            print(f"  Peak: {snap['max_allocated'] / 1e9:.2f} GB")


# Usage in trainer
class DebuggableTrainer(DreamTrainer):
    def training_step(self, batch, batch_idx):
        self.memory_tracker.snapshot(f"step_{batch_idx}_start")

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        self.memory_tracker.snapshot(f"step_{batch_idx}_after_data")

        logits = self.model(x)
        self.memory_tracker.snapshot(f"step_{batch_idx}_after_forward")

        loss = F.cross_entropy(logits, y)
        self.backward(loss)
        self.memory_tracker.snapshot(f"step_{batch_idx}_after_backward")

        if not self.is_accumulating_gradients:
            self.step(self.optimizer)
            self.memory_tracker.snapshot(f"step_{batch_idx}_after_step")

        return {"loss": loss}
```

### Memory Snapshot for Debugging OOM

```python
def enable_memory_history():
    """Enable detailed memory history for OOM debugging."""
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
    )


def dump_memory_snapshot(filename="memory_snapshot.pickle"):
    """Dump memory snapshot for analysis."""
    torch.cuda.memory._dump_snapshot(filename)


def analyze_memory_snapshot(filename):
    """Analyze memory snapshot."""
    import pickle

    with open(filename, 'rb') as f:
        snapshot = pickle.load(f)

    # Print top allocations
    allocations = sorted(
        snapshot['allocations'],
        key=lambda x: x['size'],
        reverse=True
    )

    print("Top 10 allocations:")
    for alloc in allocations[:10]:
        print(f"  {alloc['size'] / 1e9:.2f} GB - {alloc['stack'][:100]}...")
```

---

## Debugging Training Issues

### Loss Not Decreasing

```python
def diagnose_training(trainer, num_steps=100):
    """Diagnose why training might not be progressing."""
    issues = []

    # Check learning rate
    lr = trainer.optimizer.param_groups[0]['lr']
    if lr == 0:
        issues.append("Learning rate is 0!")
    elif lr > 0.1:
        issues.append(f"Learning rate {lr} may be too high")
    elif lr < 1e-7:
        issues.append(f"Learning rate {lr} may be too low")

    # Check gradients
    total_grad_norm = 0
    zero_grad_params = []

    for name, param in trainer.model.named_parameters():
        if param.grad is None:
            zero_grad_params.append(name)
        else:
            total_grad_norm += param.grad.norm().item() ** 2

    total_grad_norm = total_grad_norm ** 0.5

    if zero_grad_params:
        issues.append(f"Parameters with no gradient: {zero_grad_params[:5]}...")

    if total_grad_norm < 1e-8:
        issues.append("Gradients are near zero - vanishing gradient problem?")
    elif total_grad_norm > 1000:
        issues.append("Gradients are very large - exploding gradient problem?")

    # Check weight updates
    initial_weights = {
        name: param.data.clone()
        for name, param in trainer.model.named_parameters()
    }

    # Run a few steps
    for _ in range(num_steps):
        trainer.training_step(next(iter(trainer.train_dataloader)), 0)

    weight_changes = []
    for name, param in trainer.model.named_parameters():
        change = (param.data - initial_weights[name]).abs().mean().item()
        weight_changes.append((name, change))

    avg_change = sum(c for _, c in weight_changes) / len(weight_changes)
    if avg_change < 1e-10:
        issues.append("Weights are not updating!")

    # Report
    print("=== Training Diagnosis ===")
    print(f"Learning rate: {lr}")
    print(f"Gradient norm: {total_grad_norm:.4f}")
    print(f"Average weight change: {avg_change:.2e}")

    if issues:
        print("\nPotential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo obvious issues detected")
```

### NaN/Inf Detection

```python
import torch


def register_nan_hooks(model):
    """Register hooks to detect NaN during forward/backward."""
    def check_nan_forward(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                raise RuntimeError(
                    f"NaN detected in forward pass of {module.__class__.__name__}"
                )

    def check_nan_backward(module, grad_input, grad_output):
        for i, grad in enumerate(grad_output):
            if grad is not None and torch.isnan(grad).any():
                raise RuntimeError(
                    f"NaN detected in backward pass of {module.__class__.__name__}"
                )

    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_forward)
        module.register_full_backward_hook(check_nan_backward)
```

---

## Performance Debugging

### Find Graph Breaks (torch.compile)

```python
from dream_trainer.callbacks import FindGraphBreaksCallback


# Use the built-in callback
config = MyConfig(
    device_parameters=DeviceParameters(compile_model=True),
    callbacks=CallbackCollection([
        FindGraphBreaksCallback(
            log_graph_breaks=True,
            fail_on_graph_break=False,
        ),
    ]),
)


# Or manually debug
def debug_compile(model):
    """Debug torch.compile graph breaks."""
    import torch._dynamo as dynamo

    # Enable verbose logging
    dynamo.config.verbose = True
    dynamo.config.log_level = logging.DEBUG

    # Compile and check
    compiled = torch.compile(model, fullgraph=False)

    # Run with example input
    example_input = torch.randn(1, 3, 224, 224).cuda()

    try:
        output = compiled(example_input)
    except Exception as e:
        print(f"Compilation failed: {e}")

    # Get compilation stats
    print(f"Graphs created: {dynamo.utils.counters['stats']['graphs']}")
    print(f"Graph breaks: {dynamo.utils.counters['stats']['graph_breaks']}")
```

### Profile Training

```python
from dream_trainer.callbacks import ProfileCallback


# Use built-in profiler
config = MyConfig(
    callbacks=CallbackCollection([
        ProfileCallback(
            profile_steps=range(10, 20),
            output_dir="./profiles",
            activities=["cpu", "cuda"],
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ),
    ]),
)


# Manual profiling
def profile_training_step(trainer, batch):
    """Profile a single training step."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        trainer.training_step(batch, 0)

    # Print table
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    # Export trace
    prof.export_chrome_trace("trace.json")
```

---

## Common Issues and Solutions

### Issue: CUDA Out of Memory

```python
# Solutions:
# 1. Reduce batch size
config = MyConfig(
    training_parameters=TrainingParameters(train_batch_size=4),  # Smaller
)

# 2. Enable gradient checkpointing
config = MyConfig(
    device_parameters=DeviceParameters(checkpoint_activations=True),
)

# 3. Use CPU offloading
config = MyConfig(
    device_parameters=DeviceParameters(cpu_offload=True),
)

# 4. Use more aggressive FSDP sharding
config = MyConfig(
    device_parameters=DeviceParameters(dp_shard=8),
)
```

### Issue: Training Hangs

```python
# Debug hangs:
# 1. Enable NCCL debug
os.environ["NCCL_DEBUG"] = "INFO"

# 2. Check for deadlocks
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# 3. Add timeout to distributed ops
config = MyConfig(
    device_parameters=DeviceParameters(
        communication_timeout=timedelta(minutes=10),
    ),
)

# 4. Use barrier debugging
def debug_barrier():
    rank = dist.get_rank()
    print(f"Rank {rank} entering barrier")
    dist.barrier()
    print(f"Rank {rank} exited barrier")
```

### Issue: Different Results Across Runs

```python
# Ensure reproducibility:
from dream_trainer.utils import seed_everything

seed_everything(42, deterministic=True)

# Also set:
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
```

---

## Debug Tools Reference

| Tool | Purpose | Command |
|------|---------|---------|
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Distributed debugging | Environment variable |
| `NCCL_DEBUG=INFO` | NCCL communication debugging | Environment variable |
| `CUDA_LAUNCH_BLOCKING=1` | Synchronous CUDA | Environment variable |
| `torch.profiler` | Performance profiling | See ProfileCallback |
| `torch.cuda.memory._record_memory_history` | Memory debugging | See memory section |
| `torch._dynamo.config.verbose=True` | Compile debugging | See compile section |

---

## Next Steps

- [Performance Guide](performance.md): Optimize training throughput
- [Troubleshooting](troubleshooting.md): Common problems and solutions
- [FAQ](faq.md): Frequently asked questions
