# Comparison

Dream Trainer is not trying to be the only good PyTorch training tool. TorchTitan, Hugging Face Accelerate, and PyTorch Lightning each solve real problems well. Dream Trainer exists for a specific middle ground: teams that want reusable distributed training infrastructure while keeping the actual training algorithm in user-owned PyTorch code.

The core idea is:

> The trainer encodes the algorithm. The config encodes the scale.

That means a trainer for a small debug model should be able to scale to a 1B+ parameter model by changing config: model size, dataloaders, `DeviceParameters`, training schedule, callbacks, checkpointing, logging, and profiling. The trainer code should not need to be rewritten just because the run moved from one GPU to DDP, FSDP, HSDP, tensor parallelism, or a multi-node launch.

## Influences

Dream Trainer is heavily inspired by [TorchTitan](https://github.com/pytorch/torchtitan) and [Lightning Fabric](https://lightning.ai/docs/fabric/latest). Both point in the right direction for modern PyTorch training:

- TorchTitan shows what a PyTorch-native distributed stack can look like when modern scaling techniques are first-class.
- Lightning Fabric recognizes that many advanced users need direct control over the training loop instead of a fully managed trainer abstraction.

Dream Trainer takes those ideas and organizes them around a different product shape: reusable trainer building blocks, explicit algorithm hooks, config-driven scale, and fast iteration across model sizes and parallelism strategies.

The goal is not to hide PyTorch or invent a second training language. The goal is to keep the trainer readable while making the repetitive distributed infrastructure reusable.

## At A Glance

| Tool | Best Fit | What It Optimizes For | Dream Trainer Difference |
| --- | --- | --- | --- |
| Dream Trainer | Advanced PyTorch teams with custom training algorithms and distributed scale needs. | User-owned training logic plus reusable lifecycle, mesh, checkpoint, callback, logging, and validation infrastructure. | `training_step` and `validation_step` are first-class algorithm code; scale changes should flow through config and explicit parallelism hooks. |
| TorchTitan | Large-scale generative AI and LLM pretraining, especially Llama-style workloads. | A PyTorch-native reference stack for modern distributed LLM training. | Dream Trainer is less model-domain-specific and more focused on reusing custom trainers across GANs, RL, multi-model systems, diffusion, and other non-standard loops. |
| Hugging Face Accelerate | Existing PyTorch scripts that need distributed launch, device placement, mixed precision, or FSDP/DeepSpeed with minimal code changes. | Keep most of a raw PyTorch loop while adapting it to different hardware setups. | Dream Trainer provides a fuller trainer lifecycle and production surface: callbacks, validation cadence, trainer state, DCP checkpoints, metrics, and config-driven parallelism. |
| Lightning Fabric | Advanced users who want to write their own loop while getting device, precision, and distributed setup helpers. | Maximum control with fewer distributed-training chores than raw PyTorch. | Dream Trainer agrees with the need for control, but provides reusable trainer building blocks so each project does not rebuild its own trainer from scratch. |
| PyTorch Lightning | Users who want a managed trainer around a `LightningModule` and a broad ecosystem of batteries-included training features. | Automating the standard training loop while exposing hooks and manual optimization for advanced cases. | Dream Trainer treats custom optimization flow as the normal path, not an escape hatch, while keeping modern distributed PyTorch mechanics visible. |

## The Main Wedge

Dream Trainer is strongest when the training step itself is part of the research or product logic.

That includes:

- GANs with alternating generator and discriminator updates
- RL with rollout, policy, critic, reward, or replay-buffer phases
- diffusion models with custom schedules or multi-objective losses
- teacher-student systems
- multi-model or multi-optimizer systems
- staged objectives and curriculum schedules
- validation logic that does more than compute a single loss
- models where FSDP, TP, CP, or PP policy is architecture-specific

In those projects, the goal is not just to "run a model faster." The goal is to preserve algorithmic control while still avoiding repeated distributed-training infrastructure work.

## Compared To TorchTitan

[TorchTitan](https://github.com/pytorch/torchtitan) is a PyTorch-native platform for training generative AI models. Its README describes it as a clean-room implementation of PyTorch-native scaling techniques, with support for Llama 3.1 pretraining, FSDP2, tensor parallelism, async TP, pipeline parallelism, context parallelism, meta-device initialization, activation checkpointing, distributed checkpointing, `torch.compile`, Float8/MXFP8, DDP, HSDP, TorchFT, checkpointable dataloading, metrics, profiling, and a Python config registry.

TorchTitan's strength is that it is a serious reference stack for modern LLM pretraining. If the workload is close to the supported generative AI path, TorchTitan gives a well-tested, high-signal starting point.

Dream Trainer's strength is different:

- It is a trainer substrate, not an LLM pretraining platform.
- It does not assume one canonical pretraining loop.
- It lets each trainer own `training_step`, `validation_step`, optimizer stepping, metrics, and model-specific parallelism hooks.
- It is meant to serve multiple trainer families in one codebase, including workloads that do not look like Llama pretraining.

TorchTitan's extension model uses concepts such as `ModelSpec`, `ModelConverter`, config registries, and trainer/config subclassing. Dream Trainer puts the primary customization point directly in the trainer subclass. The lifecycle is reusable, but the algorithm remains plain code.

Use TorchTitan when you want a strong LLM pretraining reference stack. Use Dream Trainer when you want to make your own trainer scalable and repeatable across model sizes, objectives, and parallelism strategies.

## Compared To Accelerate

[Hugging Face Accelerate](https://huggingface.co/docs/accelerate/quicktour) is excellent when you already have a PyTorch script and want to run it across different distributed setups with minimal edits. Its quick tour emphasizes `accelerate launch`, `Accelerator.prepare`, and `accelerator.backward` as a small set of changes that adapt a PyTorch loop to multiple GPUs, TPUs, mixed precision, and related distributed setups.

That is a different layer than Dream Trainer.

Accelerate helps adapt a loop. Dream Trainer gives you a reusable loop architecture.

With Accelerate, the script still typically owns the surrounding lifecycle: validation cadence, callback dispatch, checkpoint state layout, metric reset/compute behavior, multi-model state, logging conventions, and production run structure. That is often exactly what a small or medium project wants.

Dream Trainer becomes useful when that repeated lifecycle code becomes the burden:

- launch and distributed world setup
- named device mesh validation
- meta-device model setup and materialization
- gradient accumulation behavior
- validation and sanity validation
- callback hooks
- DCP checkpoint state
- logger and metric integration
- profiling, benchmarking, graph-break inspection, and fault-tolerance callbacks

Use Accelerate when you want to keep a raw script and add distributed adaptation. Use Dream Trainer when you want the training code to become a reusable system without surrendering the step logic.

## Compared To Lightning Fabric

[Lightning Fabric](https://lightning.ai/docs/fabric/latest) is one of Dream Trainer's closest philosophical influences. Fabric is positioned as a flexible alternative to a fully fledged trainer: it lets users write their own training and inference logic down to individual optimizer calls, and it does not force a standardized epoch-based loop like Lightning Trainer.

That diagnosis is right. Advanced workloads often need control over the loop.

The tradeoff is that Fabric gives you lower-level building blocks, but the project still has to assemble and maintain its own trainer architecture. That is powerful for one script. It becomes repetitive when a team has many trainers that all need the same lifecycle ideas: setup ordering, validation cadence, callbacks, checkpoint state, metric reset/compute behavior, logging conventions, gradient accumulation, distributed state, profiling, and fault tolerance.

Dream Trainer sits one layer above Fabric's philosophy:

- Fabric says: keep control and write your own trainer.
- Dream Trainer says: keep control, but do not rewrite the trainer infrastructure every time.

In Dream Trainer, the reusable pieces are already organized as trainer lifecycle, mixins, callbacks, config dataclasses, and device-mesh-aware setup. The custom algorithm still lives in plain methods like `training_step`, `validation_step`, `configure_optimizers`, and model-specific parallelism hooks.

Use Fabric when you want lightweight distributed primitives around a custom loop. Use Dream Trainer when you want those primitives organized into reusable trainer building blocks for fast iteration and repeated experimentation.

## Compared To Lightning

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html) organizes code around `LightningModule` and a `Trainer` that automates loop details such as gradient enablement, dataloader execution, callbacks, device placement, backward, and optimizer steps. Lightning also has [manual optimization](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html) for advanced research topics such as reinforcement learning, sparse coding, and GANs.

Lightning's strength is a polished managed trainer. It is a good fit when the Lightning abstraction matches the project and the team wants a broad ecosystem of trainer features.

The historical pressure that led to Fabric is also the pressure Dream Trainer is designed around: many advanced users wanted less constraint than `LightningModule` and Lightning Trainer could comfortably provide, but still wanted structure and distributed support. Lightning Fabric answers by dropping down to lower-level primitives. Dream Trainer answers by keeping a trainer abstraction, but making the trainer user-owned, explicit, and config-driven.

Dream Trainer starts from a different assumption: advanced users may want the trainer to own the algorithmic control flow directly.

For example, in Dream Trainer a multi-optimizer trainer can make the update schedule explicit:

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

That code is not outside the framework. It is the intended framework surface. Dream Trainer wraps it with distributed lifecycle, callbacks, logging, validation, checkpointing, and state management.

Use Lightning when you want a managed trainer and a familiar high-level ecosystem. Use Dream Trainer when custom optimization, multiple phases, model-specific distributed policy, and config-driven scale are central to the project.

## Config-Driven Scaling

Dream Trainer is designed so the same trainer can run at different scales.

Debug run:

```python
config = MyTrainerConfig(
    model=SmallModelConfig(...),
    device_parameters=DeviceParameters.SINGLE_DEVICE(compile_model=False),
    training_parameters=TrainingParameters(
        n_epochs=1,
        train_steps_per_epoch=20,
        val_steps_per_epoch=4,
        num_sanity_val_steps=2,
    ),
)
```

Scale-up run:

```python
config = MyTrainerConfig(
    model=LargeModelConfig(...),
    device_parameters=DeviceParameters.FSDP(
        tensor_parallel="auto",
        dp_shard="auto",
        compile_model=True,
        compiled_autograd=True,
    ),
    training_parameters=TrainingParameters(
        n_epochs=...,
        train_steps_per_epoch=...,
        val_steps_per_epoch=...,
        gradient_accumulation_steps=...,
    ),
    callbacks=callbacks.CallbackCollection(
        [
            callbacks.LoggerCallback(log_every_n_train_batches=20),
            callbacks.ProgressBar(metric="train/loss"),
            callbacks.AsyncCheckpointCallback(checkpoint_parameters),
        ]
    ),
)
```

The trainer still owns the model setup, training step, validation step, optimizer mapping, and parallelism hooks. The config changes the execution environment and production features.

## Decision Guide

Choose Dream Trainer when:

- the algorithmic training logic is custom
- the same trainer needs to run from small debug to large distributed scale
- config-driven development is more important than one-off script adaptation
- FSDP, DDP, TP, CP, PP, checkpointing, logging, metrics, profiling, or fault tolerance need to compose
- the team wants plain PyTorch model and optimizer code to remain inspectable
- you want Fabric-like control without rebuilding the same trainer infrastructure for every experiment

Choose TorchTitan when:

- the workload is LLM/generative AI pretraining
- the supported TorchTitan model/config path is close to what you need
- you want a PyTorch-native reference implementation with public large-scale LLM proof points

Choose Accelerate when:

- you have a raw PyTorch script and want minimal changes
- distributed launch, placement, mixed precision, or FSDP/DeepSpeed adaptation is the main need
- you prefer to keep the lifecycle in the script

Choose Lightning Fabric when:

- you want low-level distributed, precision, and device helpers
- you are comfortable owning the whole trainer loop yourself
- you are building one-off or highly specialized training scripts where framework lifecycle is more burden than help

Choose Lightning when:

- you want a mature managed trainer
- the `LightningModule` structure fits the project
- standard loop automation and ecosystem integrations matter more than owning every training phase directly

## Summary

Dream Trainer's niche is not "more magic." It is durable structure around explicit PyTorch.

You write the trainer once to express the algorithm. You change config to change model size, data, scale, parallelism, checkpointing, logging, and debugging behavior. That is what makes it especially useful for training systems where the loop itself is part of the work.
