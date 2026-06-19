# Distributed World API

Modules:

- `dream_trainer.trainer.world.distributed_world`
- `dream_trainer.trainer.world.fault_tolerant_world`

World utilities build and expose distributed process groups, device meshes, ranks, and mesh views.

## Public Classes

| Class | Purpose |
| --- | --- |
| `DistributedWorld` | Standard distributed world and device mesh wrapper. |
| `FaultTolerantWorld` | Fault-tolerant world integration when optional dependencies are enabled. |

## Mesh Dimensions

`DistributedWorld` builds named mesh dimensions in this order:

1. `pp`
2. `dp_replicate`
3. `dp_shard`
4. `cp`
5. `tp`

It also exposes useful flattened mesh views for data-parallel, context-parallel, tensor-parallel, and sharded combinations.

## Trainer Usage

Use the trainer's `world` attribute inside hooks:

```python
def configure_dataloaders(self):
    return self.config.train_data.initialize(
        rank=self.world.dp_rank,
        world_size=self.world.dp_size,
    )
```

Avoid deriving data-parallel behavior from global rank when the mesh has multiple dimensions.

## See It In Use

- [Core Concepts — The Device Mesh](../../core-concepts.md#the-device-mesh) — what the dimension names mean and why the flattened views exist.
- [Tutorial 2 — Rank-Aware Dataloaders](../../tutorials/multi-gpu.md#3-rank-aware-dataloaders) — `self.world.dp_rank` in a real dataloader.
- [Parallelism](../../parallelism.md) — how `DeviceParameters` dimensions map to world fields.
