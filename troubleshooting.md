# Troubleshooting

Use this page when a Dream Trainer run fails before, during, or after training. For debugging workflow and optimizer-level symptoms, also read [Debugging](debugging.md).

## Import Fails

Symptom:

```text
ModuleNotFoundError: No module named 'torchmetrics'
ModuleNotFoundError: No module named 'wandb'
```

Cause: the current top-level `DreamTrainer` import composes metric and WandB mixins, so those optional packages are needed for the default trainer path.

Fix:

```bash
pip install "dream-trainer[metrics,wandb]"
```

## Single-Device Run Launches More Than One Rank

Symptom: mesh validation fails or the process group starts multiple local ranks even though the config uses `DeviceParameters.SINGLE_DEVICE()`.

Cause: `entrypoint` launches one process per visible CUDA device when no distributed environment already exists.

Fix:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

For multi-GPU runs, switch the device config to DDP, FSDP, or HSDP.

## Async Tensor Parallelism Requires Compile

Symptom:

```text
Async tensor parallelism requires model compilation
```

Cause: `async_tensor_parallel=True` while `compile_model=False`.

Fix for single-device:

```python
DeviceParameters.SINGLE_DEVICE(compile_model=False)
```

Fix for FSDP debugging:

```python
DeviceParameters.FSDP(
    compile_model=False,
    async_tensor_parallel=False,
)
```

## Missing Parallelism Hook

Symptom: setup fails after enabling DDP, FSDP, tensor parallelism, context parallelism, pipeline parallelism, activation checkpointing, or compile.

Cause: Dream Trainer owns lifecycle ordering, but the trainer must supply model-specific policy.

Fix: implement the hook that matches the enabled feature:

| Enabled Feature | Hook |
| --- | --- |
| DDP replication | `apply_replicate` |
| FSDP sharding | `apply_fully_shard` |
| Tensor parallelism | `apply_tensor_parallel` |
| Context parallelism | `apply_context_parallel` |
| Pipeline parallelism | `apply_pipeline_parallel` |
| Activation checkpointing | `apply_activation_checkpointing` |
| Model compile | `apply_compile` |

## Meta Tensors Remain

Symptom:

```text
Invalid model state after setup; meta tensors remain after materialization
```

Fixes:

- Create modules in `configure_models`.
- Initialize or load weights in `init_weights`.
- Do not replace modules after materialization unless you also place and initialize them.
- Check custom buffers and lazily-created parameters.

## Dataloader Length Errors

Symptom: setup cannot infer epoch length, or iterable datasets behave inconsistently.

Fixes:

- Set `train_steps_per_epoch` for streaming or iterable train data.
- Set `val_steps_per_epoch` for streaming or iterable validation data.
- Make dataloaders rank-aware with `self.world.dp_rank` and `self.world.dp_size`.

## Checkpoint Does Not Resume

Common causes:

- `model_state_dict` keys changed.
- The optimizer-to-model mapping changed.
- `resume_data=True` is trying to restore incompatible dataloader state.
- `strict_load=True` is enabled while the model structure changed intentionally.

Fixes:

- Keep checkpoint keys stable.
- Use `resume_data=False` when changing datasets.
- Use `LoadPartialCheckpointCallback` for weights-only initialization.
- Use `strict_load=False` when intentionally loading partial state.

## MkDocs Build Fails

Fixes:

- Run `mkdocs build --strict` from `dream-trainer/`.
- Keep nav entries limited to files that exist.
- Keep theme configuration within the `mkdocs-shadcn` options in `mkdocs.yml`.
- Avoid adding mkdocstrings blocks unless the docs environment can import Dream Trainer and its optional dependencies.
- Check Markdown whitespace with `git diff --check`.

## Quick Triage Checklist

- [ ] Reproduce on one visible GPU.
- [ ] Disable compile when possible.
- [ ] Run sanity validation.
- [ ] Confirm dataloader batch shapes.
- [ ] Confirm `self.backward(loss)` runs.
- [ ] Confirm optimizer parameters require gradients.
- [ ] Confirm returned log keys match checkpoint monitors.
- [ ] Confirm distributed world size matches device mesh dimensions.
