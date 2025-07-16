# World Management

The world management utilities handle distributed training context, device meshes, and fault tolerance. The `DistributedWorld` class is the core abstraction for managing distributed training environments.

## DistributedWorld

The main class for managing distributed training context:

::: dream_trainer.trainer.world.DistributedWorld
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - setup
        - get_mesh
        - is_global_zero
        - all_reduce
        - all_gather
        - barrier
        - get_total_norm
        - train_context
        - loss_parallel
        - get_fsdp_config
        - get_fsdp_mp_policy

### Properties

- `world_size` - Total number of processes
- `rank` - Current process rank
- `local_rank` - Rank within the node
- `device` - Current device (cuda:X or cpu)
- `dp_size` - Data parallel world size
- `dp_rank` - Data parallel rank
- `tp_enabled` - Whether tensor parallelism is enabled
- `pp_enabled` - Whether pipeline parallelism is enabled
- `cp_enabled` - Whether context parallelism is enabled

## Usage Examples

### Basic Usage

```python
from dream_trainer.trainer.world import DistributedWorld
from dream_trainer.configs import DeviceParameters

# Create world for FSDP training
device_params = DeviceParameters.FSDP()
world = DistributedWorld(device_params)

# Setup distributed environment
world.setup()

# Access world properties
print(f"World size: {world.world_size}")
print(f"Current rank: {world.rank}")
print(f"Device: {world.device}")
```

### Collective Operations

```python
# All-reduce a tensor across all processes
tensor = torch.tensor([world.rank], device=world.device)
reduced = world.all_reduce(tensor, op="sum")
# reduced will be sum of all ranks

# All-gather tensors from all processes
gathered = world.all_gather(tensor)
# gathered will contain tensors from all ranks

# Synchronize all processes
world.barrier()
```

### Device Meshes

The world manages device meshes for different parallelism strategies:

```python
# Get specific mesh
tp_mesh = world.get_mesh("tp")  # Tensor parallel mesh
dp_mesh = world.get_mesh("dp_shard")  # Data parallel shard mesh

# Check if parallelism is enabled
if world.tp_enabled:
    print("Using tensor parallelism")

if world.dp_shard_enabled:
    print("Using FSDP sharding")
```

### Training Context

```python
# Use the training context for mixed precision
with world.train_context():
    # Training code here - automatically handles autocast
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Loss parallelism for TP
with world.loss_parallel():
    # Loss computation with proper TP handling
    loss = parallel_cross_entropy(logits, labels)
```

## FaultTolerantWorld

Extended world for fault-tolerant training with torchft:

::: dream_trainer.trainer.world.FaultTolerantWorld
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false

### Usage

```python
from dream_trainer.trainer.world import FaultTolerantWorld
from dream_trainer.configs import FaultToleranceParameters

ft_params = FaultToleranceParameters(
    enable=True,
    min_replica_size=2,
    max_consecutive_failures=3
)

world = FaultTolerantWorld(device_params, ft_params)
world.setup()

# Access fault tolerance manager
ft_manager = world.ft_manager
```

## FSDP Configuration

The world provides FSDP configuration based on device parameters:

```python
# Get FSDP configuration
fsdp_config = world.get_fsdp_config()

# Configuration includes:
# - Sharding strategy
# - CPU offloading
# - Auto-wrap policy
# - Backward prefetch
# - Forward prefetch

# Get mixed precision policy
mp_policy = world.get_fsdp_mp_policy()
# Configures param_dtype and reduce_dtype
```

## Gradient Operations

```python
# Compute total gradient norm across distributed parameters
parameters = model.parameters()
total_norm = world.get_total_norm(
    parameters=parameters,
    norm_type=2.0,
    error_if_nonfinite=True
)

print(f"Gradient norm: {total_norm}")
```

## Advanced Features

### Custom Mesh Construction

```python
# The world automatically constructs meshes based on config
# Mesh dimensions: [pp, cp, dp_replicate, dp_shard, tp]

# Example 4D mesh for HSDP + TP
# pp=1, cp=1, dp_replicate=2, dp_shard=4, tp=8
# Total world size = 1 * 1 * 2 * 4 * 8 = 64
```

### Context Parallelism

```python
if world.cp_enabled:
    cp_mesh = world.get_mesh("cp")
    # Use for sequence parallelism
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
```

### Pipeline Parallelism

```python
if world.pp_enabled:
    pp_mesh = world.get_mesh("pp")
    pp_rank = pp_mesh.get_local_rank()
    
    # Check if this rank has first/last stage
    has_first_stage = (pp_rank == 0)
    has_last_stage = (pp_rank == pp_mesh.size() - 1)
```

## Integration with Trainer

The world is automatically created and managed by trainers:

```python
class MyTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        # self.world is automatically created
        
    def training_step(self, batch, batch_idx):
        # Use world for distributed operations
        if self.world.is_global_zero:
            print("Running on main process")
        
        # Collective operations
        loss = self.compute_loss(batch)
        avg_loss = self.world.all_reduce(loss, op="mean")
        
        return {"loss": avg_loss}
```

## Best Practices

### 1. Check Rank for I/O

```python
# Only perform I/O on rank 0
if world.is_global_zero:
    save_checkpoint(model)
    write_logs(metrics)
```

### 2. Use Appropriate Contexts

```python
# Training with autocast
with world.train_context():
    loss = training_step(batch)

# Loss computation with TP
if world.loss_parallel_enabled:
    with world.loss_parallel():
        loss = compute_loss(logits, labels)
```

### 3. Handle Device Placement

```python
# Always use world.device for tensors
tensor = torch.randn(10, 10, device=world.device)

# Move model to device
model = model.to(world.device)
```

### 4. Synchronize When Needed

```python
# Ensure all processes are ready
world.barrier()

# Synchronize metrics
metric = compute_metric()
synced_metric = world.all_reduce(metric, op="mean")
```

## Troubleshooting

### Common Issues

1. **Timeout errors**: Increase `comm.init_timeout_seconds`
2. **NCCL errors**: Check network configuration
3. **Device mismatch**: Ensure all tensors use `world.device`
4. **Rank mismatch**: Verify WORLD_SIZE and RANK env vars

### Debugging

```python
# Print debug info
logger.info(f"Rank {world.rank}/{world.world_size}")
logger.info(f"Device: {world.device}")
logger.info(f"DP size: {world.dp_size}, TP size: {world.get_mesh('tp').size()}")

# Check mesh configuration
logger.info(f"Mesh dims: {world.world_mesh.mesh_dim_names}")
```

## See Also

- [DeviceParameters](../configuration/parameters.md#deviceparameters) - Parallelism configuration
- [AbstractTrainer](../trainers/abstract.md) - Trainer integration
- [Parallelism Guide](../../parallelism.md) - Parallelism strategies 