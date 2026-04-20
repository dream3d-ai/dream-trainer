# Debugging

!!! abstract "TL;DR"
    - Most failures trace to a **lifecycle phase** — launch, configure, setup, train, validate, checkpoint, logging.
    - First-run checklist: **one GPU**, `compile_model=False`, `num_sanity_val_steps=2`, `ProgressBar`, `LoggerCallback(log_every_n_train_batches=1)`.
    - Meta tensors remaining after setup → move weight init into `init_weights`, not `configure_models`.
    - No gradients → `self.backward(loss)` not called, loss detached, or optimizer owns unused parameters.
    - Use `FindGraphBreaksCallback` when compiled regions misbehave — or run `python train.py find-graph-breaks` from the CLI.

Dream Trainer keeps model-specific decisions explicit, which makes most failures traceable to a single lifecycle phase. This page is a symptom-first reference.

## Start with sanity validation

Set a small sanity validation count while developing a trainer:

```python
TrainingParameters(num_sanity_val_steps=2)
```

!!! tip "Why this matters"
    Sanity validation catches dataloader, device-movement, metric, and validation-shape issues before a long training run starts. Two steps is enough to catch 80% of the bugs that would otherwise surface an hour into training.

## Symptoms and fixes

### Meta tensors remain after setup

```text
Invalid model state after setup; meta tensors remain after materialization
```

**Cause:** a parameter or buffer was never materialized, or a module was created outside the normal setup path.

!!! danger "Init order matters"
    `init_weights` runs **after** materialization. Initializing weights in `configure_models` silently discards them against meta tensors.

Fixes:

- Create modules in `configure_models`.
- Initialize or load weights in `init_weights`, **not** `configure_models`.
- Avoid replacing modules after materialization unless you also place and initialize them correctly.
- Check custom modules for buffers that don't implement reset behavior.

### `apply_compile` is missing

```text
Please implement compile_model or set device_parameters.compile_model=False
```

**Cause:** `DeviceParameters.compile_model=True` and the trainer doesn't implement `apply_compile`.

=== "Quick fix (first runs)"

    ```python
    DeviceParameters.SINGLE_DEVICE(compile_model=False)
    ```

=== "Production fix"

    ```python
    def apply_compile(self):
        self.model.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
    ```

### Wrong number of visible GPUs

**Symptom:** single-device runs fail during mesh validation or distributed launch.

**Cause:** `entrypoint` launches one process per visible GPU, but `DeviceParameters.SINGLE_DEVICE()` expects one rank.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

For multi-GPU runs, use a distributed preset (`DDP()`, `FSDP()`, `HSDP(...)`) and implement the required parallelism hook.

### No gradients or zero gradients

```text
Tried to step model with no gradients on any of its parameters
All optimizer gradients are exactly zero before optimizer step
```

**Causes & fixes:**

| Cause | Fix |
| --- | --- |
| `self.backward(loss)` not called | Call `self.backward(loss)` inside `training_step`. |
| Loss detached from model parameters | Check `.detach()` / `torch.no_grad()` around the forward path. |
| Optimizer owns parameters unused in forward | For DDP, audit replication policy and drop unused parameters. |
| All parameters are frozen | Unfreeze at least one trainable parameter. |
| Stepped during accumulation | Step only when `not self.is_accumulating_gradients`. |

### Non-finite parameters or gradients

```text
Invalid gradients before optimizer step
Gradient norm is non-finite before optimizer step
```

Fixes:

- Log the loss **before** backward.
- Reduce learning rate.
- Enable gradient clipping with `gradient_clip_val`.
- Check dtype-sensitive operations under bf16.
- Validate input data ranges and masks.
- Temporarily disable compile so stack traces are readable.

### Dataloader length mismatch

**Symptoms:** fewer batches than expected, global step-count agreement failures, iterable-dataset length errors.

Fixes:

- Set `train_steps_per_epoch` and `val_steps_per_epoch` for iterable datasets.
- Make dataloaders rank-aware with `self.world.dp_rank` and `self.world.dp_size`.
- Ensure batch size is large enough for the number of workers and data-parallel ranks.

### Checkpoint resume fails

| Cause | Fix |
| --- | --- |
| `model_state_dict` keys don't match current model | Keep state-dict key names stable across refactors. |
| Optimizer-to-model mapping changed | Re-check `configure_optimizers` return value. |
| `resume_data=True` after dataset change | Set `resume_data=False` for intentional data changes. |
| `strict_load=True` with intentional model changes | Use `strict_load=False` or `LoadPartialCheckpointCallback`. |

See [Checkpointing](checkpointing.md) for the full state layout.

## Finding graph breaks

Use `FindGraphBreaksCallback` when compiled regions behave unexpectedly:

```python
callbacks.CallbackCollection(
    [
        callbacks.FindGraphBreaksCallback(log_file="graph_breaks.log"),
    ]
)
```

The callback disables normal compilation during setup, recompiles selected functions under Dynamo explanation, and writes graph-break information to the log file.

!!! tip "Ad-hoc graph-break inspection"
    If `[cli]` is installed, you can run this without editing the trainer: `python train.py find-graph-breaks`. See [Using the CLI](cli.md).

## First-run checklist

- [ ] Run one visible GPU first.
- [ ] Set `compile_model=False` until the trainer is correct.
- [ ] Use `num_sanity_val_steps=2`.
- [ ] Return scalar `train/loss` and `val/loss` logs.
- [ ] Confirm dataloader batch shapes before model forward.
- [ ] Confirm `self.backward(loss)` runs before `self.step`.
- [ ] Confirm optimizer parameters require gradients.
- [ ] Add `ProgressBar(metric="train/loss")`.
- [ ] Add `LoggerCallback(log_every_n_train_batches=1)` for early runs.

## Next steps

- [Performance](performance.md) — once the trainer is correct, re-enable compile and distributed optimizations one at a time.
- [Using the CLI](cli.md) — ad-hoc diagnostic subcommands.
