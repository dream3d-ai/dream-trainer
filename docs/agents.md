# AI Agent Friendly

Dream Trainer is designed to be easy for humans and AI coding agents to inspect, modify, and scale. The algorithm lives in explicit trainer hooks, scale lives in config, and production behavior lives in callbacks.

That separation matters for code generation. An agent can make smaller, more local changes because the codebase has stable places for different kinds of training behavior.

## What Agent-Friendly Means Here

Agent-friendly does not mean distributed training becomes simple. It means the library gives an AI coding agent enough structure to reason about where a change belongs.

In a hand-rolled training script, model setup, dataloader construction, distributed launch, loss computation, checkpointing, validation, metrics, logging, optimizer stepping, and profiling often live in one long control flow. That makes generated edits risky: a small request can require touching code that also affects unrelated behavior.

Dream Trainer splits those concerns into explicit surfaces:

| Change | Stable Edit Surface |
| --- | --- |
| Change model structure | `configure_models` |
| Initialize or load weights | `init_weights` |
| Change optimizer ownership | `configure_optimizers` |
| Change scheduler ownership | `configure_schedulers` |
| Change data input | `configure_dataloaders` |
| Change training algorithm | `training_step` |
| Change evaluation behavior | `validation_step` and `configure_metrics` |
| Change checkpointable model state | `model_state_dict` |
| Add logging, profiling, checkpointing, or progress | callbacks |
| Change distributed scale | `DeviceParameters` |
| Change model-specific parallelism | `apply_replicate`, `apply_fully_shard`, `apply_tensor_parallel`, `apply_pipeline_parallel`, `apply_activation_checkpointing`, `apply_compile` |

That map is useful to people, and it is especially useful to agents.

## Config Is A Safe Control Plane

Agents are strongest when the desired change is expressed through structured data instead of broad rewrites. Dream Trainer makes many common experiment changes config-only:

```python
DeviceParameters.SINGLE_DEVICE(compile_model=False)
DeviceParameters.DDP()
DeviceParameters.FSDP()
DeviceParameters.HSDP(dp_shard=8)
```

The same applies to training length, validation cadence, gradient accumulation, logging, checkpointing, metrics, and callback stacks.

For example, an agent can change a debug run into a larger FSDP run by editing config:

```python
config = MyTrainerConfig(
    model=LargeModelConfig(...),
    device_parameters=DeviceParameters.FSDP(
        tensor_parallel="auto",
        dp_shard="auto",
        compile_model=True,
    ),
    training_parameters=TrainingParameters(
        train_steps_per_epoch=...,
        val_steps_per_epoch=...,
        gradient_accumulation_steps=...,
    ),
)
```

The trainer's algorithm does not move. The execution strategy changes around it.

## Plain PyTorch Is Still Readable

Dream Trainer keeps the training algorithm in ordinary Python and PyTorch. That is important because agents already understand common PyTorch idioms.

For a GAN-style trainer, the core control flow can stay visible:

```python
def training_step(self, batch, batch_idx):
    if self.should_update_discriminator(batch_idx):
        loss = self.discriminator_loss(batch)
        self.backward(loss)
        logs = {"train/discriminator_loss": loss}
        if not self.is_accumulating_gradients:
            logs["train/discriminator_grad_norm"] = self.step(
                self.discriminator_optimizer
            )
        return logs

    loss = self.generator_loss(batch)
    self.backward(loss)
    logs = {"train/generator_loss": loss}
    if not self.is_accumulating_gradients:
        logs["train/generator_grad_norm"] = self.step(self.generator_optimizer)
    return logs
```

An agent can inspect that code and reason about the update schedule directly. It does not need to infer algorithmic behavior from a large framework config or hidden trainer loop.

## Building Blocks Reduce Blast Radius

Callbacks make production behavior additive. An agent can add a checkpoint, logger, profiler, progress bar, graph-break inspector, EMA callback, FP8 callback, FSDP optimizer, or fault-tolerance callback without editing the loss function.

```python
callbacks=callbacks.CallbackCollection(
    [
        callbacks.LoggerCallback(log_every_n_train_batches=20),
        callbacks.ProgressBar(metric="train/loss"),
        callbacks.AsyncCheckpointCallback(checkpoint_parameters),
        callbacks.ProfileCallback(...),
    ]
)
```

That is a smaller and safer change than threading new side effects through a custom training loop.

Mixins play a similar role. They keep reusable trainer behavior separated from model-specific algorithm code, so agents can reason about lifecycle features without treating every trainer as a unique one-off script.

## Reviewability Is Part Of The Design

Agent-generated changes should be easy to review. Dream Trainer helps because a reviewer can ask whether the edit touched the right layer:

- Did a scale change stay in config?
- Did a model architecture change stay in `configure_models`?
- Did weight loading happen in `init_weights` rather than under the meta-device path?
- Did algorithm changes stay in `training_step` or `validation_step`?
- Did logging, checkpointing, profiling, or progress use callbacks?
- Did distributed policy go into an explicit parallelism hook?

This gives reviewers a practical checklist instead of forcing them to read an entire training script for every change.

## Useful Failure Modes Help Agents Iterate

Agents work better when errors are concrete. Dream Trainer's structure creates failures that usually point to a lifecycle phase:

- a missing `apply_compile` hook when compile is enabled
- a missing `apply_fully_shard` or `apply_replicate` hook when FSDP or DDP is enabled
- meta tensors left after materialization
- a device mesh whose dimensions do not match world size
- dataloader length mismatches
- checkpoint state keys that do not match the current trainer
- gradients that are missing, zero, or non-finite before an optimizer step

Those errors are easier for an agent to diagnose than generic failures in a deeply interleaved training script.

## Good Agent Tasks

Dream Trainer is especially well-suited for agent tasks like:

- add a new callback to an existing trainer
- convert a single-device config into an FSDP config
- add sanity validation and debug logging
- add DCP checkpointing with a `val/loss` monitor
- split a quick-start trainer into `config.py` and `train.py`
- add a second optimizer and make the update schedule explicit
- add metrics without changing the training loss
- add an `apply_fully_shard` hook for a model with repeated blocks
- add a non-compiled debug config next to a compiled production config
- compare two configs without changing trainer code

These tasks have narrow edit surfaces and clear acceptance checks.

## Tasks That Still Need Care

Some requests remain hard even with structure:

- inventing a correct tensor-parallel plan for an unfamiliar architecture
- changing the mathematical objective of an RL algorithm
- modifying pipeline-parallel schedules
- debugging numerical instability at scale
- proving convergence behavior
- tuning performance across a multi-node cluster

Dream Trainer does not remove the need for expert review. It makes the generated changes easier to localize, run, and inspect.

## Guidance For Agents

When using an AI agent to modify Dream Trainer code, give it the intended layer:

- "Change config only."
- "Do not modify `training_step`."
- "Add checkpointing through callbacks."
- "Keep the trainer algorithm unchanged; make this run as FSDP."
- "Add a new optimizer branch in `training_step` and update `model_state_dict` if needed."
- "Add a debug config with `compile_model=False`."

Good prompts map cleanly onto Dream Trainer's structure. That is the point: the framework gives agents and humans the same set of rails.

Dream Trainer is agent-friendly because it combines plain PyTorch with stable architecture. The model and algorithm remain visible, while repetitive infrastructure is organized into configs, hooks, mixins, callbacks, and checkpoint state.

That makes fast iteration safer: an AI agent can help change how a trainer scales, logs, checkpoints, validates, or profiles without rewriting the algorithm. And when the algorithm does need to change, the relevant code is explicit enough for both the agent and the reviewer to reason about.
