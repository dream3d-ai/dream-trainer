# Performance Callbacks

Dream Trainer provides callbacks for profiling, optimizing, and debugging training performance.

## ProfileCallback

Profile training performance using PyTorch's profiler:

::: dream_trainer.callbacks.ProfileCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - pre_fit

### Usage

```python
import torch.profiler
from dream_trainer.callbacks import ProfileCallback

# Create profiler
profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=2
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)

# Add to callbacks
callback = ProfileCallback(profiler)
```

### Analyzing Results

```python
# View in TensorBoard
# tensorboard --logdir=./logs

# Or analyze programmatically
profiler.export_chrome_trace("trace.json")
print(profiler.key_averages().table(sort_by="cuda_time_total"))
```

## FindGraphBreaksCallback

Debug torch.compile graph breaks:

::: dream_trainer.callbacks.FindGraphBreaksCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - pre_launch
        - pre_train_step

### Usage

```python
from dream_trainer.callbacks import FindGraphBreaksCallback

# Find all graph breaks
graph_breaks = FindGraphBreaksCallback(
    log_file="graph_breaks.log",
    skip=0,  # Skip no steps
    fullgraph=False  # Test without fullgraph
)

# Test if model can compile with fullgraph=True
fullgraph_test = FindGraphBreaksCallback(
    log_file="fullgraph_test.log",
    skip=0,
    fullgraph=True
)
```

### Example Output

```
================================================================================
model.encoder.attention
/path/to/model.py:45 - Graph break due to:
  - Dynamic control flow (if statement depending on tensor value)
  - Suggested fix: Use torch.where or masked operations

================================================================================
model.decoder.generate
/path/to/model.py:123 - Graph break due to:
  - Python builtin not supported in graph
  - Suggested fix: Use torch operations instead of Python list comprehension
```

## OptimizeFSDP

Optimize FSDP performance with advanced prefetching:

::: dream_trainer.callbacks.OptimizeFSDP
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - pre_train_step
        - post_train_step

### Usage

```python
from dream_trainer.callbacks import OptimizeFSDP

# Basic FSDP optimization
fsdp_opt = OptimizeFSDP(
    prefetch=1,  # Prefetch 1 module ahead
    display=False
)

# Aggressive prefetching
fsdp_opt = OptimizeFSDP(
    prefetch=2,  # Prefetch 2 modules ahead
    display=True  # Show execution order
)
```

### How It Works

1. **Traces execution order** during first training step
2. **Sets up prefetching** based on traced order
3. **Optimizes overlap** between computation and communication

Benefits:
- 10-20% speedup for large models
- Better GPU utilization
- Reduced communication overhead

## FaultToleranceCallback

Enable fault-tolerant training with torchft:

::: dream_trainer.callbacks.FaultToleranceCallback
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false

### Usage

```python
from dream_trainer.callbacks import FaultToleranceCallback
from dream_trainer.configs import FaultToleranceParameters

ft_config = FaultToleranceParameters(
    checkpoint_interval=100,  # Checkpoint every 100 steps
    max_failures=3,          # Tolerate up to 3 failures
    heartbeat_interval=60,   # Check health every 60 seconds
)

ft_callback = FaultToleranceCallback(ft_config)
```

## Fp8Quantization

Enable FP8 training for memory and compute efficiency:

::: dream_trainer.callbacks.Fp8Quantization
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false

### Usage

```python
from dream_trainer.callbacks import Fp8Quantization

# Enable FP8 training
fp8 = Fp8Quantization(
    enabled=True,
    amax_history_len=1024,  # History for scaling factors
    amax_compute_algo="most_recent"  # or "max"
)
```

## Creating Custom Performance Callbacks

### Throughput Monitor

```python
class ThroughputMonitor(Callback[BaseTrainer]):
    """Monitor training throughput."""
    
    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.start_time = None
        self.samples_processed = 0
    
    def pre_train_step(self, batch: dict[str, Any], batch_idx: int):
        if batch_idx == self.warmup_steps:
            self.start_time = time.time()
            self.samples_processed = 0
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        if batch_idx > self.warmup_steps and self.start_time is not None:
            batch_size = self.trainer.config.batch_size
            self.samples_processed += batch_size
            
            # Log every 100 steps
            if batch_idx % 100 == 0:
                elapsed = time.time() - self.start_time
                throughput = self.samples_processed / elapsed
                
                print(f"Throughput: {throughput:.2f} samples/sec")
                
                if hasattr(self.trainer, "log_scalar"):
                    self.trainer.log_scalar("perf/throughput", throughput)
```

### Compilation Monitor

```python
class CompilationMonitor(Callback[BaseTrainer]):
    """Monitor torch.compile performance."""
    
    def __init__(self):
        self.compile_times = {}
        self.graph_breaks = {}
    
    def pre_setup(self):
        # Hook into torch.compile
        original_compile = torch.compile
        
        def monitored_compile(model, *args, **kwargs):
            start = time.time()
            compiled = original_compile(model, *args, **kwargs)
            elapsed = time.time() - start
            
            model_name = model.__class__.__name__
            self.compile_times[model_name] = elapsed
            
            print(f"Compiled {model_name} in {elapsed:.2f}s")
            return compiled
        
        torch.compile = monitored_compile
    
    def post_setup(self):
        # Report compilation summary
        total_time = sum(self.compile_times.values())
        print(f"\nCompilation Summary:")
        print(f"Total time: {total_time:.2f}s")
        
        for name, time in self.compile_times.items():
            print(f"- {name}: {time:.2f}s")
```

### Memory Profiler

```python
class MemoryProfiler(Callback[BaseTrainer]):
    """Profile memory usage patterns."""
    
    def __init__(self, profile_every: int = 100):
        self.profile_every = profile_every
        self.memory_stats = []
    
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        if batch_idx % self.profile_every == 0:
            if torch.cuda.is_available():
                # Get memory stats
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                
                # Get peak stats
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
                
                stats = {
                    "step": batch_idx,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "peak_allocated_gb": peak_allocated,
                    "peak_reserved_gb": peak_reserved,
                    "fragmentation": (reserved - allocated) / reserved
                }
                
                self.memory_stats.append(stats)
                
                # Log if fragmentation is high
                if stats["fragmentation"] > 0.3:
                    print(f"⚠️ High memory fragmentation: {stats['fragmentation']:.1%}")
    
    def post_fit(self):
        # Save detailed report
        import json
        with open("memory_profile.json", "w") as f:
            json.dump(self.memory_stats, f, indent=2)
```

## Performance Optimization Strategies

### 1. Profiling Strategy

```python
# Profile different phases
callbacks = CallbackCollection([
    # Warmup without profiling
    WarmupCallback(steps=100),
    
    # Profile training
    ProfileCallback(
        torch.profiler.profile(
            activities=[ProfilerActivity.CUDA],
            schedule=schedule(wait=100, warmup=10, active=20)
        )
    ),
    
    # Monitor throughput
    ThroughputMonitor(warmup_steps=100)
])
```

### 2. FSDP Optimization

```python
# Combine FSDP optimizations
callbacks = CallbackCollection([
    # Optimize prefetching
    OptimizeFSDP(prefetch=2),
    
    # Monitor memory
    MemoryProfiler(profile_every=50),
    
    # Track throughput
    ThroughputMonitor()
])
```

### 3. Debugging Compilation

```python
# Debug torch.compile issues
callbacks = CallbackCollection([
    # Find graph breaks
    FindGraphBreaksCallback("breaks.log"),
    
    # Monitor compilation
    CompilationMonitor(),
    
    # Profile compiled vs eager
    ProfileCallback(profiler)
])
```

## Analyzing Performance

### Using TensorBoard

```python
# Launch TensorBoard
# tensorboard --logdir=./logs

# View:
# - GPU utilization
# - Kernel timings
# - Memory usage
# - Operator breakdown
```

### Creating Reports

```python
class PerformanceReporter(Callback[BaseTrainer]):
    """Generate performance reports."""
    
    def __init__(self):
        self.metrics = {
            "throughput": [],
            "gpu_util": [],
            "memory": [],
            "compile_time": 0
        }
    
    def post_fit(self):
        # Generate report
        report = f"""
Performance Report
==================

Training Summary:
- Total time: {self.total_time:.2f}s
- Average throughput: {np.mean(self.metrics['throughput']):.2f} samples/s
- Peak memory: {max(self.metrics['memory']):.2f} GB
- Compilation time: {self.metrics['compile_time']:.2f}s

GPU Utilization:
- Average: {np.mean(self.metrics['gpu_util']):.1f}%
- Min: {min(self.metrics['gpu_util']):.1f}%
- Max: {max(self.metrics['gpu_util']):.1f}%

Recommendations:
{self.get_recommendations()}
"""
        
        with open("performance_report.txt", "w") as f:
            f.write(report)
```

## Best Practices

### 1. Profile in Stages

```python
# Don't profile everything at once
# Stage 1: Data loading
# Stage 2: Forward pass
# Stage 3: Backward pass
# Stage 4: Optimizer step
```

### 2. Use Appropriate Tools

- **CPU bottlenecks**: cProfile, py-spy
- **GPU bottlenecks**: nsight, PyTorch profiler
- **Memory issues**: memory_profiler, tracemalloc
- **Compilation**: torch._dynamo.explain

### 3. Automate Analysis

```python
class AutoAnalyzer(Callback[BaseTrainer]):
    """Automatically identify bottlenecks."""
    
    def analyze_profile(self, profile_data):
        # Find slow operations
        slow_ops = self.find_slow_operations(profile_data)
        
        # Detect patterns
        if self.is_communication_bound(profile_data):
            print("Training is communication bound")
            print("Consider: larger batch size, gradient accumulation")
        
        if self.is_memory_bound(profile_data):
            print("Training is memory bound")
            print("Consider: activation checkpointing, CPU offload")
```

## See Also

- [Callback System](base.md) - Base callback documentation
- [Optimization Guide](../../performance.md) - Performance tuning
- [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [torch.compile Guide](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) 