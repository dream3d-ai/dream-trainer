# Getting Started with Dream Trainer

Welcome to Dream Trainer! This guide will help you understand what makes Dream Trainer unique and get you training models with advanced parallelism in minutes.

## Why Dream Trainer?

Before we dive in, let's understand what makes Dream Trainer different:

### 🧩 **Composable Mixin Architecture**
Unlike monolithic frameworks, Dream Trainer lets you compose exactly the features you need:

```python
# Minimal trainer - just the essentials
class SimpleTrainer(BaseTrainer, SetupMixin):
    pass

# Add features as needed
class ProductionTrainer(BaseTrainer, SetupMixin, WandBLoggerMixin, 
                       EvalMetricMixin, QuantizeMixin):
    pass  # Now with logging, metrics, and quantization!
```

### 🚀 **DTensor-Native from Day One**
Every parameter in Dream Trainer is a DTensor, giving you:
- Automatic support for new PyTorch sharding patterns
- Clean, debuggable distributed code
- First-class support for TP, PP, CP, and FSDP2

### ⚡ **Zero-Compromise Performance**
- Intelligent FSDP prefetching that traces execution order
- Loss parallelism for tensor-parallel training
- Async tensor parallelism support
- Compiled autograd integration

### 📝 **Configs as Code for Type Safety**
Dream Trainer embraces Python configs over YAML/JSON for better developer experience:

```python
# ❌ Traditional approach - error-prone strings
config = {
    "model": "gpt2",  # Typo? Wrong name? Who knows!
    "lr": "3e-4",     # String or float? 
    "layers": 12,     # Is this valid for gpt2?
}

# ✅ Dream Trainer - full type safety and IDE support
@dataclass
class MyConfig(BaseTrainerConfig):
    learning_rate: float = 3e-4  # Type-checked!
    num_layers: int = 12         # Auto-completion!
    
    def validate(self):
        """Custom validation logic"""
        if self.num_layers < 1:
            raise ValueError("Need at least 1 layer!")
```

Benefits of configs as code:
- **Type Safety**: Catch config errors at definition time, not runtime
- **IDE Support**: Auto-completion, refactoring, and go-to-definition
- **Composability**: Use functions, inheritance, and composition
- **Validation**: Add custom validation logic and constraints
- **Documentation**: Docstrings and type hints document themselves

```python
# Example: Composable configs with validation
def make_model_config(size: Literal["small", "base", "large"]) -> ModelConfig:
    """Factory function for common model sizes"""
    sizes = {
        "small": ModelConfig(hidden_size=768, num_layers=12),
        "base": ModelConfig(hidden_size=1024, num_layers=24),
        "large": ModelConfig(hidden_size=1536, num_layers=48),
    }
    return sizes[size]

# Use partial functions for complex configs
from functools import partial

config = TrainerConfig(
    model=partial(TransformerModel, num_heads=16),
    optimizer=partial(torch.optim.AdamW, betas=(0.9, 0.95)),
    scheduler=make_cosine_scheduler(warmup_steps=1000),
)
```

## Installation

To get started with Dream Trainer:

```bash
pip install dream-trainer
```

For detailed installation instructions, including:
- System requirements and CUDA compatibility
- Feature-specific installations (wandb, metrics, quantization, etc.)
- Development setup
- Docker and cluster deployments
- Troubleshooting common issues

Please see our comprehensive [Installation Guide](installation.md).

## Your First Trainer

Let's build a trainer that showcases Dream Trainer's strengths:

### Step 1: Understanding the Mixin Pattern

Dream Trainer uses mixins to compose functionality. Here's the anatomy:

```python
from dataclasses import dataclass
from dream_trainer import BaseTrainer, BaseTrainerConfig
from dream_trainer.trainer.mixins import SetupMixin, SetupConfigMixin

# 1. Configuration uses the same mixin pattern
@dataclass
class MyTrainerConfig(BaseTrainerConfig, SetupConfigMixin):
    # BaseTrainerConfig provides: epochs, batch_size, etc.
    # SetupConfigMixin adds: model/optimizer/dataloader configs
    learning_rate: float = 3e-4
    hidden_size: int = 768
    
# 2. Trainer mirrors the config structure
class MyTrainer(BaseTrainer, SetupMixin):
    config: MyTrainerConfig
    
    # SetupMixin requires these methods:
    def configure_models(self):
        """Define your models (on meta device - no memory used!)"""
        self.model = TransformerModel(self.config.hidden_size)
    
    def init_weights(self):
        """Initialize weights after parallelism is applied"""
        self.model.apply(self._init_weights)
    
    def configure_optimizers(self):
        """Define optimizers"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def configure_dataloaders(self):
        """Return train and validation dataloaders"""
        return self._make_train_loader(), self._make_val_loader()
    
    # BaseTrainer requires these methods:
    def training_step(self, batch, batch_idx):
        """Your forward pass and loss computation"""
        loss = self.model(batch)
        self.backward(loss)  # Handles gradient accumulation!
        
        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.model, self.optimizer)
            return {"loss": loss, "grad_norm": grad_norm}
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation forward pass"""
        with torch.no_grad():
            loss = self.model(batch)
        return {"val_loss": loss}
```

### Step 2: Add Advanced Parallelism

Here's where Dream Trainer shines - adding parallelism is simple:

```python
from dream_trainer.configs import DeviceParameters

# Configure parallelism declaratively
config = MyTrainerConfig(
    device_parameters=DeviceParameters(
        # Data parallelism
        dp_shard=4,           # FSDP2 across 4 devices
        dp_replicate=2,       # DDP across 2 nodes (HSDP)
        
        # Model parallelism  
        tensor_parallel=4,    # Tensor parallel degree
        pipeline_parallel=2,  # Pipeline stages
        context_parallel=2,   # For long sequences
        
        # Optimizations
        async_tensor_parallel=True,
        compile_model=True,
        loss_parallel=True,
    )
)

# That's it! Dream Trainer handles all the complexity
```

### Step 3: Implement Parallelism Methods

For advanced parallelism, implement these methods in your trainer:

```python
class MyTrainer(BaseTrainer, SetupMixin):
    # ... previous methods ...
    
    def apply_tensor_parallel(self, tp_mesh):
        """Apply tensor parallelism to your model"""
        # Dream Trainer provides the mesh, you decide the sharding
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
        
        # Parallelize attention layers
        for layer in self.model.layers:
            tp_plan = {
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(), 
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(),
            }
            parallelize_module(layer, tp_mesh, tp_plan)
    
    def apply_pipeline_parallel(self, pp_mesh):
        """Split model into pipeline stages"""
        # Return pipeline schedule and split modules
        stages = [
            self.model.embed,
            self.model.layers[:8],
            self.model.layers[8:16],
            self.model.output
        ]
        
        from torch.distributed.pipelining import pipeline_parallel
        schedule = pipeline_parallel(stages, pp_mesh)
        
        return {"model": (schedule, stages, True, True)}
    
    def apply_fully_shard(self, fsdp_config):
        """Apply FSDP2 sharding"""
        from torch.distributed._composable.fsdp import fully_shard
        
        # Shard each transformer layer
        for layer in self.model.layers:
            fully_shard(layer, **fsdp_config)
        
        # Shard the whole model
        fully_shard(self.model, **fsdp_config)
```

## Complete Example: Multi-GPU Language Model

Let's put it all together with a realistic example:

```python
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.callbacks import (
    LoggerCallback, 
    CheckpointCallback,
    OptimizeFSDP,
    CallbackCollection
)
from dream_trainer.configs import (
    DeviceParameters,
    CheckpointParameters,
    TrainingParameters
)

@dataclass
class LMConfig(DreamTrainerConfig):
    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    # Training
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    
    # Data
    sequence_length: int = 2048
    dataset_path: str = "data/openwebtext"

class LanguageModelTrainer(DreamTrainer):
    config: LMConfig
    
    def configure_models(self):
        """Models are created on meta device - no memory used!"""
        from my_models import GPTModel
        
        self.model = GPTModel(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
        )
    
    def init_weights(self):
        """Initialize after parallelism is applied"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        self.model.apply(_init_weights)
    
    def configure_optimizers(self):
        """Configure AdamW with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.config.learning_rate)
    
    def configure_schedulers(self):
        """Cosine schedule with warmup"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training_parameters.n_epochs,
            eta_min=self.config.learning_rate * 0.1
        )
    
    def configure_dataloaders(self):
        """Create distributed dataloaders"""
        from my_data import TextDataset
        
        train_dataset = TextDataset(
            self.config.dataset_path,
            sequence_length=self.config.sequence_length,
            split='train'
        )
        
        # Dream Trainer provides distributed sampling utilities
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_parameters.train_batch_size,
            sampler=self.get_train_sampler(train_dataset),  # Handles DP/PP
            num_workers=4,
            pin_memory=True
        )
        
        val_dataset = TextDataset(
            self.config.dataset_path,
            sequence_length=self.config.sequence_length,
            split='validation'
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_parameters.train_batch_size * 2,
            sampler=self.get_val_sampler(val_dataset),
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def training_step(self, batch, batch_idx):
        """Forward pass with next-token prediction"""
        input_ids = batch['input_ids']
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Loss computation with optional loss parallelism
        with self.loss_parallel():
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Backward handles gradient accumulation automatically
        self.backward(loss)
        
        # Step optimizer only when not accumulating
        if not self.is_accumulating_gradients:
            grad_norm = self.step(self.model, self.optimizer)
            
            return {
                "loss": loss,
                "grad_norm": grad_norm,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """Compute validation perplexity"""
        input_ids = batch['input_ids']
        
        with torch.no_grad():
            logits = self.model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            perplexity = torch.exp(loss)
        
        return {
            "val_loss": loss,
            "val_perplexity": perplexity,
        }
    
    def apply_tensor_parallel(self, tp_mesh):
        """Apply tensor parallelism to transformer layers"""
        from torch.distributed.tensor.parallel import (
            ColwiseParallel, 
            RowwiseParallel,
            PrepareModuleInput,
            parallelize_module
        )
        
        # Parallelize each transformer block
        for i, block in enumerate(self.model.blocks):
            layer_plan = {
                # Attention
                "attn.q_proj": ColwiseParallel(),
                "attn.k_proj": ColwiseParallel(), 
                "attn.v_proj": ColwiseParallel(),
                "attn.out_proj": RowwiseParallel(),
                
                # MLP
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            }
            
            parallelize_module(
                block,
                tp_mesh,
                layer_plan,
                input_fn=PrepareModuleInput(),
            )

# Create configuration with advanced features
config = LMConfig(
    # Distributed settings
    device_parameters=DeviceParameters(
        dp_shard=4,              # 4-way FSDP2
        tensor_parallel=2,       # 2-way tensor parallelism
        compile_model=True,      # torch.compile
        enable_compiled_autograd=True,
        loss_parallel=True,      # Parallel loss computation
    ),
    
    # Training settings
    training_parameters=TrainingParameters(
        n_epochs=10,
        train_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch = 32
        gradient_clip_val=1.0,
        val_frequency=0.25,  # Validate 4x per epoch
    ),
    
    # Callbacks for production features
    callbacks=CallbackCollection([
        LoggerCallback(log_every_n_train_batches=10),
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_every_n_epochs=1,
                keep_top_k=3,
                monitor="val_perplexity",
            )
        ),
        OptimizeFSDP(prefetch=2),  # Intelligent FSDP prefetching
    ])
)

if __name__ == "__main__":
    # Dream Trainer handles distributed launch automatically
    from dream_trainer.utils import distributed_launch
    
    def main():
        trainer = LanguageModelTrainer(config)
        trainer.fit()
    
    distributed_launch(main)
```

## Launch Training

### Single GPU
```bash
python train.py
```

### Multiple GPUs (Single Node)
```bash
# Dream Trainer auto-detects available GPUs
python train.py

# Or explicitly with torchrun
torchrun --nproc_per_node=8 train.py
```

### Multiple Nodes
```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=29500 train.py

# Node 1  
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_ADDR --master_port=29500 train.py
```

## Understanding Dream Trainer's Advantages

### 1. **Clean Parallelism Abstractions**

Dream Trainer makes complex parallelism approachable:

```python
# Bad (Raw PyTorch)
if rank == 0:
    model = model.cuda(0)
    # Complex manual sharding...
    
# Good (Dream Trainer)
def apply_tensor_parallel(self, tp_mesh):
    # Clean, reusable parallelism logic
```

### 2. **Automatic Mixed Precision**

Dream Trainer wraps forward methods intelligently:

```python
# Automatic - Dream Trainer handles autocast placement
def training_step(self, batch, batch_idx):
    loss = self.model(batch)  # Autocast applied automatically
    
# No need for manual autocast contexts!
```

### 3. **Gradient Accumulation That Just Works**

```python
# Dream Trainer handles the complexity
self.backward(loss)  # Scales by accumulation steps

if not self.is_accumulating_gradients:
    # Only step when ready
    self.step(self.model, self.optimizer)
```

### 4. **Composable Features**

Add features without modifying your core trainer:

```python
# Start simple
class V1Trainer(BaseTrainer, SetupMixin):
    pass

# Add metrics later
class V2Trainer(BaseTrainer, SetupMixin, EvalMetricMixin):
    def configure_metrics(self):
        self.accuracy = Accuracy()

# Add logging even later
class V3Trainer(BaseTrainer, SetupMixin, EvalMetricMixin, WandBLoggerMixin):
    pass  # No changes to existing code!
```

## Common Patterns

### Adding Custom Callbacks

```python
from dream_trainer.callbacks import Callback

class LearningRateWarmup(Callback[SetupMixin]):
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
    
    def post_train_step(self, result, batch_idx):
        if self.trainer.global_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.trainer.global_step / self.warmup_steps
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale
```

### Debugging Distributed Training

```python
# Dream Trainer provides utilities for distributed debugging
from dream_trainer.utils import rank_zero_print

class MyTrainer(DreamTrainer):
    def training_step(self, batch, batch_idx):
        # Only prints from rank 0
        rank_zero_print(f"Batch shape: {batch['input_ids'].shape}")
        
        # Check DTensor sharding
        if hasattr(self.model.weight, 'placements'):
            rank_zero_print(f"Weight sharding: {self.model.weight.placements}")
```

## Next Steps

Now that you understand Dream Trainer's core concepts:

1. **Explore Mixins**: Check out the [Trainer Guide](trainer-guide.md) to see all available mixins
2. **Master Parallelism**: Read the [Parallelism Guide](parallelism.md) for advanced distributed training
3. **Extend with Callbacks**: Learn to create custom callbacks in the [Callbacks Guide](callbacks.md)
4. **Optimize Performance**: See [Best Practices](best-practices.md) for performance tips

## Troubleshooting

### Installation Issues

```bash
# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"

# Verify Dream Trainer features
python -c "from dream_trainer.utils import check_features; check_features()"
```

### Common Issues

1. **OOM with Large Models**: Enable CPU offloading or use gradient checkpointing
2. **Slow Data Loading**: Increase `num_workers` and use `pin_memory=True`
3. **Debugging Distributed**: Set `TORCH_CPP_LOG_LEVEL=INFO` for detailed logs

## Getting Help

- 📚 [Full Documentation](index.md)
- 💬 [GitHub Discussions](https://github.com/dream-trainer/dream-trainer/discussions)
- 🐛 [Issue Tracker](https://github.com/dream-trainer/dream-trainer/issues)
- 💡 [Examples Repository](https://github.com/dream-trainer/dream-trainer/tree/main/examples)

---

**Ready to train with Dream Trainer?** You now understand what makes it unique. Happy training! 🚀
