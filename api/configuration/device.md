# Device Parameters API

Module: `dream_trainer.configs.trainer`

`DeviceParameters` describes mixed precision, compile behavior, communication timeouts, and named distributed mesh dimensions.

## Public Classes

| Class | Purpose |
| --- | --- |
| `DeviceParameters` | Main distributed device and mesh configuration. |
| `Comm` | Communication timeout and flight-recorder settings. |

## Presets

```python
DeviceParameters.SINGLE_DEVICE()
DeviceParameters.DDP()
DeviceParameters.FSDP()
DeviceParameters.HSDP(dp_shard=8)
```

## Mesh Dimensions

| Field | Mesh Dimension |
| --- | --- |
| `_pipeline_parallel` | `pp` |
| `_dp_replicate` | `dp_replicate` |
| `_dp_shard` | `dp_shard` |
| `_context_parallel` | `cp` |
| `_tensor_parallel` | `tp` |

At most one dimension can be `"auto"`. Otherwise, the product of all dimensions must match world size.

## Compile And Async Tensor Parallelism

`async_tensor_parallel=True` requires `compile_model=True`. For non-compiled debug runs, set both:

```python
DeviceParameters.FSDP(
    compile_model=False,
    async_tensor_parallel=False,
)
```

`SINGLE_DEVICE(compile_model=False)` already disables async tensor parallelism.

## See It In Use

- [Parallelism — Pick A Mode First](../../parallelism.md#pick-a-mode-first) — decision tree for picking a preset.
- [Core Concepts — The Device Mesh](../../core-concepts.md#the-device-mesh) — why the dimensions are named.
- [Tutorial 2 — New Config Variant](../../tutorials/multi-gpu.md#1-new-config-variant) — `DDP()` in a real config.
