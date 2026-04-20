---
title: Trainer Guide
---

# Trainer Guide

<small>🛠️ How-to · hook-by-hook reference</small>

!!! abstract "TL;DR"
    - **Always** implement 7 hooks: `configure_models`, `init_weights`, `model_state_dict`, `configure_optimizers`, `configure_dataloaders`, `training_step`, `validation_step`.
    - **Add when you scale**: `apply_replicate`, `apply_fully_shard`, `apply_tensor_parallel`, `apply_pipeline_parallel`, `apply_compile`, `apply_activation_checkpointing`.
    - Multi-model or frozen-module runs return a `dict[nn.Module, Optimizer]` and override `parameters_to_train`.

Every Dream Trainer trainer is a subclass of `DreamTrainer` that implements some number of hooks. The lifecycle runs the same regardless of how many you override — the difference between a 50-line single-GPU trainer and a 300-line production trainer is just how many hooks the latter implements.

This page groups hooks into two tiers: the ones you **always** write (for any trainer to work at all) and the ones you add **when you scale** (when you turn on parallelism, compile, activation checkpointing, or multi-model training). If you're writing your first trainer, read only the first section. Come back for the second when you hit the wall that requires it.

## The Seven Hooks You Always Write

Any trainer needs these. They are the contract between your model and Dream Trainer's lifecycle.

### 1. `configure_models` — module structure on meta

Create your model. This runs under a meta-device context, so the tensors are shape-only.

```python
def configure_models(self):
    self.model = MyModel(self.config.model)
```

!!! danger "Don't load weights here"
    The tensors don't exist yet. Loading weights into meta tensors silently discards them. Weight loading goes in `init_weights`.

### 2. `init_weights` — initialize real tensors

By the time this runs, Dream Trainer has applied parallelism hooks and materialized tensors on the training device. This is where you initialize parameters or load pretrained weights.

```python
def init_weights(self):
    self.model.init_weights()
```

For a straight-forward case this might just call into your model's own init. For pretraining-from-scratch, this is where `nn.init.xavier_uniform_`, `nn.init.trunc_normal_`, or your custom init lives.

### 3. `model_state_dict` — what to checkpoint

Return a DCP-compatible state dict. Use `get_model_state_dict` even in single-GPU runs — it costs nothing and keeps your trainer ready for sharded state.

```python
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict


def model_state_dict(self, **_):
    return {"model": get_model_state_dict(self.model, options=StateDictOptions())}
```

For multiple models, return a dict with one entry per model:

```python
def model_state_dict(self, **_):
    return {
        "encoder": get_model_state_dict(self.encoder),
        "decoder": get_model_state_dict(self.decoder),
    }
```

### 4. `configure_optimizers` — who optimizes what

Create optimizers and return a mapping from **model** to optimizer. The mapping is how Dream Trainer associates optimizer state with the right model during checkpoint save/load.

```python
def configure_optimizers(self):
    self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.config.optimizer.learning_rate,
    )
    return {self.model: self.optimizer}
```

!!! warning "Return the mapping"
    Returning a bare optimizer (without the model key) means Dream Trainer can't resume its state cleanly. Always return `{self.model: self.optimizer}` even for single-model trainers.

### 5. `configure_dataloaders` — rank-aware data

Return `(train_loader, val_loader)`. In any distributed run, pass `self.world.dp_rank` and `self.world.dp_size` into your dataloader factory so each data-parallel rank sees a different shard.

```python
def configure_dataloaders(self):
    return (
        self.config.train_data.build_dataloader(
            rank=self.world.dp_rank, world_size=self.world.dp_size,
        ),
        self.config.val_data.build_dataloader(
            rank=self.world.dp_rank, world_size=self.world.dp_size,
        ),
    )
```

### 6. `training_step` — one step of the loop

Compute the loss, call `self.backward(loss)`, and step the optimizer once accumulation is complete. Return a dict of scalars for logging.

```python
def training_step(self, batch, batch_idx):
    output = self.model(batch["x"])
    loss = self.loss(output, batch["y"])
    self.backward(loss)

    logs = {"train/loss": loss}
    if not self.is_accumulating_gradients:
        logs["train/grad_norm"] = self.step(self.optimizer)
    return logs
```

`self.step(optimizer)` is more than `optimizer.step()` — it validates gradients, computes and clips the gradient norm (if configured), dispatches optimizer callbacks, steps the scheduler if one is attached, and zeros gradients. Use it instead of calling `optimizer.step()` directly.

!!! warning "Three hook methods that replace their PyTorch equivalents"
    - `self.backward(loss)` instead of `loss.backward()` — handles accumulation scaling.
    - `self.step(optimizer)` instead of `optimizer.step()` — integrates clipping, callbacks, scheduler stepping.
    - `self.is_accumulating_gradients` — gates the optimizer step during accumulation.

    If you skip any of these, gradient accumulation silently misbehaves and grad norms are wrong.

### 7. `validation_step` — compute metrics

Run inference, update metrics, and return scalar logs. Wrap in `@torch.no_grad()`.

```python
@torch.no_grad()
def validation_step(self, batch, batch_idx):
    output = self.model(batch["x"])
    self.metrics.update(output, batch["y"])
    return {"val/loss": self.loss(output, batch["y"])}
```

Dream Trainer resets metrics before the validation epoch and computes them after. Your `validation_step` only needs to call `.update()`.

### Small but sometimes needed

A few more hooks you might override in the "always write" tier if your trainer needs them:

| Hook | When you need it |
| --- | --- |
| `configure_schedulers` | You want a learning-rate schedule. Return `{optimizer: scheduler}`. |
| `configure_metrics` | You compute validation metrics. Assign `self.metrics = self.config.metrics`. |

## The Hooks You Add When You Scale

Dream Trainer only calls these when the corresponding `DeviceParameters` option is enabled. For a single-GPU trainer with `compile_model=False`, you do not need any of them.

| Hook | Called when | What to implement |
| --- | --- | --- |
| `apply_compile` | `device_parameters.compile_model=True` | Compile modules or selected methods. |
| `apply_replicate` | DDP replication active (`dp_replicate > 1`) | Composable `replicate()` over the model. |
| `apply_fully_shard` | FSDP sharding active (`dp_shard > 1`) | Call `fully_shard` on layers and the root module. |
| `apply_tensor_parallel` | Tensor parallelism active (`tensor_parallel > 1`) | Apply a tensor-parallel plan to layers. |
| `apply_pipeline_parallel` | Pipeline parallelism active (`pipeline_parallel > 1`) | Split the model into stages and schedule microbatches. |
| `apply_activation_checkpointing` | `checkpoint_activations=True` | Wrap selected modules with activation checkpointing. |

Dream Trainer owns the **order** these run. You own the **policy**. See [Parallelism](parallelism.md) for hook-by-hook guidance on each mode.

### Minimal apply hooks

=== "Compile"

    ```python
    def apply_compile(self):
        self.model.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
    ```

=== "DDP"

    ```python
    from torch.distributed._composable.replicate import replicate

    def apply_replicate(self, dp_replicate_mesh):
        replicate(self.model, device_mesh=dp_replicate_mesh)
    ```

=== "FSDP"

    ```python
    from torch.distributed.fsdp import fully_shard

    def apply_fully_shard(self, config):
        for layer in self.model.layers:
            fully_shard(layer, **config)
        fully_shard(self.model, **config)
    ```

=== "TP"

    ```python
    from torch.distributed.tensor.parallel import parallelize_module

    def apply_tensor_parallel(self, tp_mesh):
        parallelize_module(self.model, tp_mesh, plan=my_tp_plan)
    ```

## Multiple Models And Frozen Modules

Dream Trainer tracks anything assigned as a trainer attribute that's an `nn.Module`. For multiple models, just assign each in `configure_models`:

```python
def configure_models(self):
    self.encoder = Encoder(self.config.encoder)
    self.decoder = Decoder(self.config.decoder)
```

Train only the decoder by mapping only its optimizer:

```python
def configure_optimizers(self):
    self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.config.lr)
    return {self.decoder: self.optimizer}
```

Frozen models still need to be initialized and placed on the right device. Set `requires_grad_(False)` and `eval()` in `init_weights` or after loading weights. They will be moved to device, but Dream Trainer won't build an optimizer for them if they don't appear in your `configure_optimizers` mapping.

## Common Mistakes

- **Loading weights in `configure_models`.** Tensors are meta there. Use `init_weights`.
- **Calling `loss.backward()` directly.** Use `self.backward(loss)` so gradient accumulation scales correctly.
- **Calling `optimizer.step()` directly.** Use `self.step(self.optimizer)` so clipping, callbacks, and scheduler stepping run.
- **Enabling compile without `apply_compile`.** Dream Trainer will tell you, but save yourself the trip — either implement the hook or set `compile_model=False`.
- **Returning an optimizer without the model mapping.** `configure_optimizers` must return `{model: optimizer}`, not a bare optimizer.
- **Forgetting rank-aware dataloaders under DP.** Every data-parallel rank sees the same batch if you don't pass `self.world.dp_rank`.

## Next Steps

- [Configuration](configuration.md) — how configs feed the hooks you just wrote.
- [Parallelism](parallelism.md) — the scaling-tier hooks in depth, with a decision tree.
- [Callbacks](callbacks.md) — what to put outside the trainer class.
