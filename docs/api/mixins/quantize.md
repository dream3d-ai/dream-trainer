# Quantization Mixins API

Module: `dream_trainer.trainer.mixins.quantize`

Quantization mixins provide trainer-side hooks and filters used by quantization callbacks.

## Public Classes

| Class | Purpose |
| --- | --- |
| `QuantizeConfigMixin` | Config support for quantization behavior. |
| `QuantizeMixin` | Trainer-side interface for applying quantization. |
| `QuantizeModuleFilter` | Base filter type for deciding which modules are quantized. |
| `ExcludeModuleByName` | Filter that excludes modules by name. |
| `ExcludeSubmodules` | Filter that excludes nested submodules. |

## Related Callback

`Fp8Quantization` integrates optional torchao FP8 quantization. Use it only when the runtime and model support the target quantization mode.

## Guidance

Keep quantization policy explicit and close to the model. Use filters for reusable inclusion or exclusion logic, and keep a non-quantized debug configuration available.

## See It In Use

- [Callbacks — Training Extension Callback API](../callbacks/training.md) — `Fp8Quantization` and `Fp8QuantizeConfig`.
- [Performance](../../performance.md) — when FP8 quantization is worth enabling.
