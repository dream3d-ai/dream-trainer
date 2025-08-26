# Dream Trainer

Dream Trainer is a powerful, distributed training framework built exclusively around PyTorch's new DTensor APIs. It provides a flexible, composable approach that makes it easy to implement and debug advanced parallelism patterns like FSDP, Tensor Parallelism, Pipeline Parallelism and Context Parallelism.

Dream Trainer was created to address these core issues:

- **Boilerplate Overload**: Each parallelism scheme requires its own verbose, error-prone setup & configuration that must be applied in the correct order. This makes it difficult to quickly switch between parallelism strategies as hardware and network topolgies change.
- **Legacy Trainer Limitations**: Most trainers are tightly coupled to old DDP/FSDP APIs and "zero-config" abstractions, making debugging harder and preventing them from taking advantage of new DTensor-based distributed patterns. Being DTensor-native makes code simpler and easier to debug.
- **Complexity in Real Workflows**: Even simple training scripts become unwieldy when mixing advanced parallelism, due to scattered configuration and framework assumptions.

Dream Trainer stands on the shoulder of giants, building on top of the PyTorch ecosystem and TorchTitan repo. We aim to to keep up to date with the latest developments tested by TorchTitan, but packaged in a way that is easy to use and understand.

## 🏗️ Design Principles

Dream Trainer is built on three core principles:

1. **Native PyTorch First**

   - Designed exclusively around PyTorch's new DTensor abstractions for simple but powerful parallelism
   - Direct integration with PyTorch's ecosystem (torchao, torchft, DCP, torchrun)

2. **Minimal Assumptions**

   - Let users make their own choices
   - No automatic model wrapping or hidden behaviors
   - Assume users know what they're doing with advanced parallelism

3. **Composable Architecture**

   - Trainer is a composition of mixins that define different parts of the training loop (setup, training, evaluation, etc.)
   - Take what you need, drop the rest. Write your own components when needed.
   - Callback system for runtime modifications to the loop

## 🌟 Key Features

### Parallelism Support

Dream Trainer provides simple configuration for all PyTorch parallelism schemes:

- **Data Parallelism**: Basic multi-GPU training with PyTorch's `replicate()` API
- **FSDP2**: Second-generation Fully Sharded Data Parallel built on DTensor
- **Tensor Parallelism (TP)**: Parameter-wise sharding via DTensor layouts; composable with FSDP2 for HSDP
- **Context Parallelism (CP)**: Sequence parallelism for extremely long contexts
- **Pipeline Parallelism (PP)**: Layer pipelining across GPUs / nodes with automatic schedule search

### Other Features via Callbacks

- **Powerful typed CLI** for runtime modifications, profiling and more.
- **Checkpointing** DCP-based checkpointing with async checkpoint support
- **Fault Tolerance** via torchft
- **Native FP8 Quantization** via torchao
- **FSDP Optimization** via graph capture

## 🤔 Why Dream Trainer vs. Other Frameworks?

While PyTorch Lightning, Accelerate and DeepSpeed simplify distributed training, they revolve around classic DDP/FSDP wrappers and hide key details behind heavyweight base classes. This gets unruly as APIs grow and evolve.

Dream Trainer takes a different path:

- **DTensor-native** from day one—every parameter is a `DTensor`, so new sharding layouts appear the moment they land in PyTorch nightly.
- **Parallel schemes (FSDP2, TP, PP, CP)** are first-class, composable primitives, not bolt-on "plugins".
- **Mix-and-match** – DreamTrainer is designed around mixins to maximize composability.
- **Minimal magic** – no metaclasses, no `LightningModule`; your model remains a plain `nn.Module`.

## 📖 Next Steps

- Follow the [Getting Started](getting-started.md) guide to install and set up Dream Trainer
- Check out the [Examples](examples/basic.md) for complete working code
- Read the [Trainer Guide](trainer-guide.md) to create your own custom trainer
