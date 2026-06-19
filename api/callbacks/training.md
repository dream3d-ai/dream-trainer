# Training Extension Callback API

Modules:

- `dream_trainer.callbacks.ema`
- `dream_trainer.callbacks.fp8`
- `dream_trainer.callbacks.ft`
- `dream_trainer.callbacks.weight_transfer`
- `dream_trainer.callbacks.model_summary`
- `dream_trainer.callbacks.trainer_summary`

Training extension callbacks add optional model or runtime behavior around the core trainer loop.

## Public Classes

| Class | Purpose |
| --- | --- |
| `EMACallback` | Maintains exponential moving averages. |
| `Fp8Quantization` | Applies optional torchao FP8 quantization. |
| `Fp8QuantizeConfig` | Configures FP8 quantization behavior. |
| `FaultToleranceCallback` | Integrates optional torchft fault-tolerance behavior. |
| `WeightTransferCallback` | Transfers weights between configured modules. |
| `ModelSummary` | Summarizes model structure when dependencies are available. |
| `TrainerSummary` | Summarizes trainer structure when dependencies are available. |

## Optional Dependencies

Some callbacks are imported only when their optional dependencies are available. Treat these as opt-in production features and keep a simpler debug configuration that does not require them.

## See It In Use

- [Callbacks — Built-in Callback Catalogue](../../callbacks.md#built-in-callback-catalogue) — a one-line summary of every shipped callback.
- [Tutorial 4 — Production Shape](../../tutorials/production.md) — `TrainerSummary` and `ModelSummary` in a real callback stack.
