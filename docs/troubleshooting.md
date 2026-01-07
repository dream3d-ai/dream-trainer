# Troubleshooting

This page covers common issues and their solutions when using Dream Trainer.

---

## Installation Issues

### PyTorch/CUDA Version Mismatch

**Symptom**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**:
```bash
# Check your CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Missing DTensor Features

**Symptom**: `AttributeError: module 'torch.distributed' has no attribute 'tensor'`

**Solution**: Upgrade PyTorch to 2.7.0+:
```bash
pip install torch>=2.7.0
```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'dream_trainer'`

**Solution**:
```bash
# Verify installation
pip show dream-trainer

# If not installed
pip install dream-trainer

# If using editable install
pip install -e .
```

---

## Memory Issues

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions** (try in order):

1. **Reduce batch size**:
```python
config = MyConfig(
    training_parameters=TrainingParameters(
        train_batch_size=4,  # Reduce from 8
        gradient_accumulation_steps=8,  # Increase to maintain effective batch
    ),
)
```

2. **Enable activation checkpointing**:
```python
config = MyConfig(
    device_parameters=DeviceParameters(
        checkpoint_activations=True,
    ),
)
```

3. **Enable CPU offloading**:
```python
config = MyConfig(
    device_parameters=DeviceParameters(
        cpu_offload=True,
    ),
)
```

4. **Increase FSDP sharding**:
```python
config = MyConfig(
    device_parameters=DeviceParameters(
        dp_shard=8,  # More sharding
    ),
)
```

5. **Use smaller model or lower precision**:
```python
config = MyConfig(
    device_parameters=DeviceParameters(
        param_dtype=torch.bfloat16,
    ),
)
```

### Memory Leak

**Symptom**: Memory usage grows over time

**Solutions**:

1. **Clear cache periodically**:
```python
from dream_trainer.callbacks import GarbageCollectionCallback

config = MyConfig(
    callbacks=CallbackCollection([
        GarbageCollectionCallback(gc_every_n_steps=100),
    ]),
)
```

2. **Don't store tensors in history**:
```python
# Bad
self.losses.append(loss)  # Stores tensor with grad history

# Good
self.losses.append(loss.item())  # Store scalar
```

3. **Detach tensors before logging**:
```python
return {"loss": loss.detach()}
```

---

## Training Issues

### Loss is NaN or Inf

**Symptom**: `loss: nan` or `loss: inf`

**Solutions**:

1. **Reduce learning rate**:
```python
config = MyConfig(learning_rate=1e-5)  # From 1e-4
```

2. **Enable gradient clipping**:
```python
config = MyConfig(
    training_parameters=TrainingParameters(
        gradient_clip_val=1.0,
    ),
)
```

3. **Check for bad data**:
```python
def training_step(self, batch, batch_idx):
    x, y = batch
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError(f"Invalid data in batch {batch_idx}")
```

4. **Add NaN detection hooks**:
```python
from dream_trainer.callbacks import DebugCallback
config = MyConfig(
    callbacks=CallbackCollection([
        DebugCallback(check_nan=True, check_inf=True),
    ]),
)
```

### Loss Not Decreasing

**Symptom**: Loss stays constant

**Solutions**:

1. **Check learning rate is not zero**:
```python
print(f"LR: {trainer.optimizer.param_groups[0]['lr']}")
```

2. **Verify gradients exist**:
```python
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")
```

3. **Check data is being shuffled**:
```python
train_loader = DataLoader(dataset, shuffle=True)
```

4. **Verify model is in training mode**:
```python
print(f"Training mode: {model.training}")
```

### Gradients are Zero

**Symptom**: All gradients are 0.0

**Solutions**:

1. **Check loss requires grad**:
```python
loss = model(x)
print(f"requires_grad: {loss.requires_grad}")  # Should be True
```

2. **Ensure optimizer has correct parameters**:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(f"Param groups: {len(optimizer.param_groups)}")
print(f"Params in first group: {len(optimizer.param_groups[0]['params'])}")
```

3. **Check for detached tensors**:
```python
# Bad - breaks gradient flow
x = x.detach()  # Don't do this before loss computation
```

---

## Distributed Issues

### Training Hangs

**Symptom**: Training stops without error

**Debugging steps**:

1. **Add rank logging**:
```python
print(f"[Rank {dist.get_rank()}] Reached step {batch_idx}")
```

2. **Enable NCCL debug**:
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

3. **Check network connectivity**:
```bash
ping <other-node>
```

4. **Verify all ranks start**:
```bash
# On each node
ps aux | grep python
```

**Common causes**:
- Different ranks have different data sizes
- Some ranks fail before reaching collective ops
- Network issues between nodes
- Firewall blocking communication

### NCCL Errors

**Symptom**: `NCCL error: unhandled system error`

**Solutions**:

1. **Check GPU topology**:
```bash
nvidia-smi topo -m
```

2. **Set correct network interface**:
```bash
export NCCL_SOCKET_IFNAME=eth0  # or your interface name
```

3. **Disable IB if not available**:
```bash
export NCCL_IB_DISABLE=1
```

4. **Increase timeout**:
```python
config = MyConfig(
    device_parameters=DeviceParameters(
        communication_timeout=timedelta(minutes=30),
    ),
)
```

### Parameter Mismatch Across Ranks

**Symptom**: `RuntimeError: Inconsistent tensor sizes`

**Solutions**:

1. **Use same random seed**:
```python
from dream_trainer.utils import seed_everything
seed_everything(42)
```

2. **Initialize model on rank 0, then broadcast**:
```python
if dist.get_rank() == 0:
    model = MyModel()
dist.broadcast(model.state_dict(), src=0)
```

3. **Check config is identical on all ranks**:
```python
config_hash = hash(str(config))
all_hashes = [None] * dist.get_world_size()
dist.all_gather_object(all_hashes, config_hash)
assert len(set(all_hashes)) == 1, "Config mismatch!"
```

---

## Checkpoint Issues

### Checkpoint Loading Fails

**Symptom**: `RuntimeError: Error loading state dict`

**Solutions**:

1. **Check key mismatch**:
```python
checkpoint = torch.load("checkpoint.pt")
model_keys = set(model.state_dict().keys())
ckpt_keys = set(checkpoint['model'].keys())

print(f"Missing in ckpt: {model_keys - ckpt_keys}")
print(f"Unexpected in ckpt: {ckpt_keys - model_keys}")
```

2. **Load with strict=False**:
```python
model.load_state_dict(checkpoint['model'], strict=False)
```

3. **Handle FSDP checkpoints**:
```python
import torch.distributed.checkpoint as dcp

# FSDP checkpoints need special loading
dcp.load(
    trainer.state_dict(),
    checkpoint_id=checkpoint_path,
)
```

### Different Results After Loading

**Symptom**: Model behaves differently after checkpoint resume

**Solutions**:

1. **Load all state (including optimizer, scheduler)**:
```python
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
scheduler.load_state_dict(ckpt['scheduler'])
trainer.global_step = ckpt['global_step']
trainer.current_epoch = ckpt['epoch']
```

2. **Reset RNG state**:
```python
torch.set_rng_state(ckpt['rng_state'])
if torch.cuda.is_available():
    torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
```

---

## Data Loading Issues

### Slow Data Loading

**Symptom**: GPU utilization is low, data loading is bottleneck

**Solutions**:

1. **Increase num_workers**:
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Increase
    pin_memory=True,
    persistent_workers=True,
)
```

2. **Use prefetching**:
```python
train_loader = DataLoader(
    dataset,
    prefetch_factor=4,
)
```

3. **Preprocess data offline**:
```python
# Save tokenized data as tensors
torch.save(tokenized_data, "preprocessed.pt")
```

### Data Mismatch Across Ranks

**Symptom**: Each rank processes different amounts of data

**Solutions**:

1. **Use DistributedSampler**:
```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
)
```

2. **Drop last incomplete batch**:
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    drop_last=True,
)
```

---

## Compilation Issues

### torch.compile Fails

**Symptom**: `torch._dynamo.exc.Unsupported`

**Solutions**:

1. **Identify the issue**:
```python
import torch._dynamo as dynamo
dynamo.config.suppress_errors = False
dynamo.config.verbose = True
```

2. **Skip problematic operations**:
```python
@torch._dynamo.disable
def problematic_function():
    ...
```

3. **Use fullgraph=False**:
```python
model = torch.compile(model, fullgraph=False)
```

4. **Check for dynamic shapes**:
```python
# Use consistent shapes or enable dynamic
model = torch.compile(model, dynamic=True)
```

---

## Getting Help

If you can't resolve your issue:

1. **Search existing issues**: [GitHub Issues](https://github.com/dream3d/dream-trainer/issues)
2. **Ask on Discord**: [Dream Trainer Discord](https://discord.gg/dream-trainer)
3. **Create a new issue** with:
   - Dream Trainer version
   - PyTorch version
   - CUDA version
   - Complete error traceback
   - Minimal reproduction code
