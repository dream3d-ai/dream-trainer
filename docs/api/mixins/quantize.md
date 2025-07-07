# Quantization Mixin

The `QuantizeMixin` provides infrastructure for model quantization during training, supporting various quantization strategies including FP8 and INT8 quantization. It enables selective quantization of model components through customizable filters.

## Overview

The `QuantizeMixin` enables:
- Selective module quantization with filters
- FP8 training support
- INT8 quantization for inference
- Custom quantization strategies
- Memory-efficient training of large models

## Class Reference

::: dream_trainer.trainer.mixins.QuantizeMixin
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __init__
        - quantized_models
        - quantize_module_filters

### Module Filters

::: dream_trainer.trainer.mixins.QuantizeModuleFilter
    options:
      show_source: true
      show_bases: false
      show_root_full_path: false
      members:
        - __call__
        - validate
        - __add__

## Built-in Filters

Dream Trainer provides several pre-built quantization filters:

### ExcludeModuleByName

::: dream_trainer.trainer.mixins.ExcludeModuleByName
    options:
      show_source: true
      show_bases: false

### ExcludeSubmodules

::: dream_trainer.trainer.mixins.ExcludeSubmodules
    options:
      show_source: true
      show_bases: false

## Usage Examples

### Basic Quantization Setup

```python
from dream_trainer.trainer.mixins import QuantizeMixin, ExcludeModuleByName
import torch.nn as nn

class MyTrainer(BaseTrainer, QuantizeMixin):
    def quantize_module_filters(self):
        # Exclude specific modules from quantization
        return {
            "model": ExcludeModuleByName([
                "model.embeddings.word_embeddings",
                "model.embeddings.position_embeddings",
                "model.lm_head"
            ])
        }
    
    def setup(self):
        super().setup()
        
        # Apply FP8 quantization
        from transformer_engine.pytorch import fp8_autocast
        
        # Models specified in quantize_module_filters will be quantized
        with fp8_autocast(enabled=True):
            # Training will use FP8 for specified modules
            pass
```

### Using ExcludeSubmodules Filter

```python
def quantize_module_filters(self):
    # Exclude entire submodule trees from quantization
    return {
        "model": ExcludeSubmodules([
            "model.embeddings",  # Excludes all embedding layers
            "model.head",        # Excludes output head
            "model.encoder.layer.0",  # Exclude first encoder layer
        ])
    }
```

### Combining Filters

Filters can be combined using the `+` operator:

```python
def quantize_module_filters(self):
    # Combine multiple filters - module must pass ALL filters
    exclude_by_name = ExcludeModuleByName([
        "model.embeddings.word_embeddings",
        "model.lm_head"
    ])
    exclude_submodules = ExcludeSubmodules([
        "model.encoder.layer.0",  # Skip first layer
        "model.encoder.layer.23"  # Skip last layer
    ])
    
    combined_filter = exclude_by_name + exclude_submodules
    
    return {"model": combined_filter}
```

### Custom Filter Implementation

Create custom filters for specific quantization strategies:

```python
class QuantizeByNamePatternFilter(QuantizeModuleFilter):
    """Quantize modules whose names match a pattern."""
    
    def __init__(self, patterns: list[str]):
        self.patterns = patterns
        self.quantized_modules = []
    
    def __call__(self, module: nn.Module, name: str) -> bool:
        # Check if module name matches any pattern
        should_quantize = any(pattern in name for pattern in self.patterns)
        
        if should_quantize:
            self.quantized_modules.append(name)
            
        return should_quantize
    
    def validate(self):
        # Ensure we quantized at least one module
        if not self.quantized_modules:
            raise ValueError(f"No modules matched patterns: {self.patterns}")
        
        print(f"Quantized {len(self.quantized_modules)} modules")

# Use the custom filter
class MyTrainer(BaseTrainer, QuantizeMixin):
    def quantize_module_filters(self):
        # Quantize all attention and MLP layers
        return {
            "model": QuantizeByNamePatternFilter([
                "attention",
                "mlp",
                "feed_forward"
            ])
        }
```

### Layer-wise Quantization

Apply different quantization strategies to different layers:

```python
class LayerWiseQuantizeFilter(QuantizeModuleFilter):
    """Apply different quantization based on layer depth."""
    
    def __init__(self, early_layers: int, late_layers: int):
        self.early_layers = early_layers
        self.late_layers = late_layers
    
    def __call__(self, module: nn.Module, name: str) -> bool:
        # Extract layer number from name
        import re
        match = re.search(r'layer\.(\d+)', name)
        
        if match:
            layer_num = int(match.group(1))
            
            # Don't quantize early or late layers
            if layer_num < self.early_layers:
                return False
            if layer_num >= self.total_layers - self.late_layers:
                return False
                
        return True

class MyTrainer(BaseTrainer, QuantizeMixin):
    def quantize_module_filters(self):
        # Skip first 2 and last 2 layers
        return {
            "model": LayerWiseQuantizeFilter(
                early_layers=2,
                late_layers=2
            )
        }
```

## FP8 Training Example

Complete example using FP8 quantization for efficient training:

```python
from dream_trainer.trainer.mixins import (
    QuantizeMixin, 
    ExcludeModuleByName,
    ExcludeSubmodules
)
import transformer_engine.pytorch as te

class FP8Trainer(BaseTrainer, QuantizeMixin):
    def configure_models(self):
        # Use TransformerEngine layers for FP8 support
        self.model = te.TransformerLayer(
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=16384,
            params_dtype=torch.float16,
            device="cuda"
        )
    
    def quantize_module_filters(self):
        # Exclude embedding and output layers from quantization
        return {
            "model": ExcludeModuleByName([
                "model.word_embeddings",
                "model.output_layer"
            ])
        }
    
    def training_step(self, batch, batch_idx):
        # Enable FP8 autocast for forward and backward
        with te.fp8_autocast(enabled=True):
            outputs = self.model(batch["input"])
            loss = self.criterion(outputs, batch["target"])
            
            # Backward pass also uses FP8
            self.backward(loss)
        
        return {"loss": loss}
```

## INT8 Quantization for Inference

Post-training quantization for deployment:

```python
class INT8QuantizationFilter(QuantizeModuleFilter):
    """Filter for INT8 quantization."""
    
    def __init__(self, calibration_batches: int = 100):
        self.calibration_batches = calibration_batches
        self.modules_to_quantize = []
    
    def __call__(self, module: nn.Module, name: str) -> bool:
        # Quantize compute-intensive layers
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if module.weight.numel() > 100_000:  # >100K parameters
                self.modules_to_quantize.append((name, module))
                return True
        return False
    
    def apply_int8_quantization(self, model):
        """Apply INT8 quantization after training."""
        import torch.quantization as quantization
        
        # Prepare model for quantization
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        
        # Calibrate with representative data
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_dataloader):
                if i >= self.calibration_batches:
                    break
                model(batch)
        
        # Convert to INT8
        quantization.convert(model, inplace=True)
        
        return model
```

## Memory-Efficient Training

Combine quantization with other memory-saving techniques:

```python
class MemoryEfficientTrainer(BaseTrainer, QuantizeMixin, SetupMixin):
    def quantize_module_filters(self):
        # Exclude critical layers from quantization
        return {
            "model": ExcludeSubmodules([
                "model.embeddings",
                "model.layer_norm",
                "model.output"
            ])
        }
    
    def apply_activation_checkpointing(self):
        # Combine with activation checkpointing
        from torch.distributed.fsdp.wrap import checkpoint_wrapper
        
        for name, module in self.model.named_modules():
            if isinstance(module, TransformerBlock):
                wrapped = checkpoint_wrapper(module)
                setattr(self.model, name, wrapped)
    
    def configure_optimizers(self):
        # Use memory-efficient optimizer
        from bitsandbytes.optim import AdamW8bit
        
        self.optimizer = AdamW8bit(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.1
        )
```

## Best Practices

### 1. Gradual Quantization

Start with less aggressive quantization and increase over time:

```python
class GradualQuantizationFilter(QuantizeModuleFilter):
    def __init__(self, start_epoch: int, full_epoch: int):
        self.start_epoch = start_epoch
        self.full_epoch = full_epoch
    
    def __call__(self, module: nn.Module, name: str) -> bool:
        if self.trainer.current_epoch < self.start_epoch:
            return False
            
        # Gradually increase quantization
        progress = (self.trainer.current_epoch - self.start_epoch) / (
            self.full_epoch - self.start_epoch
        )
        progress = min(1.0, max(0.0, progress))
        
        # Quantize based on module size and progress
        size_threshold = 1_000_000 * (1 - progress) + 100_000 * progress
        return module.weight.numel() > size_threshold
```

### 2. Validation Without Quantization

Validate model performance without quantization:

```python
def validation_step(self, batch, batch_idx):
    # Temporarily disable quantization for validation
    with torch.autocast(device_type='cuda', enabled=False):
        outputs = self.model(batch["input"])
        loss = self.criterion(outputs, batch["target"])
    
    return {"val_loss": loss}
```

### 3. Monitor Quantization Impact

Track the effect of quantization on model performance:

```python
def on_train_epoch_end(self):
    # Log quantization statistics
    total_params = sum(p.numel() for p in self.model.parameters())
    quantized_params = sum(
        p.numel() for name, p in self.model.named_parameters()
        if name in self.quantized_modules()
    )
    
    self.log_scalar(
        "quantization/param_ratio",
        quantized_params / total_params
    )
    
    # Compare FP32 vs quantized performance
    self.evaluate_quantization_impact()
```

## Troubleshooting

### Common Issues

1. **Gradient Underflow**: Use gradient scaling with FP8/INT8
2. **Accuracy Degradation**: Start with less aggressive quantization
3. **Incompatible Layers**: Not all layers support all quantization types
4. **Memory Savings**: Ensure quantization is applied before materialization

### Debugging

```python
def validate_quantization(self):
    """Ensure quantization is applied correctly."""
    for model_name in self.quantized_models():
        model = getattr(self, model_name)
        
        # Check which modules are quantized
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                dtype = module.weight.dtype
                print(f"{name}: {dtype}")
```

## See Also

- [SetupMixin](setup.md) - Model setup and parallelism
- [BaseTrainer](../trainers/base.md) - Core training functionality
- [FP8 Training Guide](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html) 