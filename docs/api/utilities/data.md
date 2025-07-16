# Data Utilities

Dream Trainer provides utilities for working with data loaders and datasets in distributed training environments.

## DataLoader Utilities

### get_epoch_length

Calculate the length of an epoch from a dataloader:

::: dream_trainer.utils.dataloader.get_epoch_length
    options:
      show_source: true

### Usage

```python
from dream_trainer.utils.dataloader import get_epoch_length

# Automatically determine length
epoch_length = get_epoch_length(train_dataloader, length=None)

# Or specify manually
epoch_length = get_epoch_length(train_dataloader, length=1000)
```

### get_train_dataloader_steps

Calculate the number of training steps:

::: dream_trainer.utils.dataloader.get_train_dataloader_steps
    options:
      show_source: true

### get_val_dataloader_steps

Calculate the number of validation steps:

::: dream_trainer.utils.dataloader.get_val_dataloader_steps
    options:
      show_source: true

## Batch Type

Dream Trainer uses a standard batch format:

```python
from dream_trainer.utils.dataloader import Batch

# Batch is a dict[str, Any]
batch: Batch = {
    "input": input_tensor,
    "target": target_tensor,
    "mask": attention_mask,
    # ... any other fields
}
```

## DataLoader Best Practices

### 1. Distributed Sampling

```python
from torch.utils.data import DataLoader, DistributedSampler

def configure_dataloaders(self):
    # Create distributed sampler for multi-GPU
    train_sampler = None
    if self.world.size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world.size,
            rank=self.world.rank,
            shuffle=True,
            seed=self.seed
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader
```

### 2. Efficient Data Loading

```python
# Use multiple workers
num_workers = min(8, os.cpu_count() // self.world.local_size)

# Pin memory for GPU transfer
pin_memory = self.world.device.type == "cuda"

# Keep workers alive
persistent_workers = num_workers > 0

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=2 if num_workers > 0 else None
)
```

### 3. Stateful DataLoaders

For resumable training with custom samplers:

```python
class StatefulDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0
    
    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "sampler_state": getattr(self.sampler, "state_dict", lambda: {})()
        }
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        if hasattr(self.sampler, "load_state_dict"):
            self.sampler.load_state_dict(state_dict["sampler_state"])
```

### 4. Memory-Efficient Loading

```python
# For large datasets
class StreamingDataset(IterableDataset):
    def __init__(self, urls: list[str], buffer_size: int = 1000):
        self.urls = urls
        self.buffer_size = buffer_size
    
    def __iter__(self):
        # Stream data from URLs
        for url in self.urls:
            yield from self.stream_from_url(url)

# Use with DataLoader
dataloader = DataLoader(
    StreamingDataset(urls),
    batch_size=None,  # Dataset yields batches
    num_workers=1     # Streaming doesn't parallelize well
)
```

## Integration with Trainer

DataLoaders are typically configured in the trainer:

```python
class MyTrainer(DataLoaderSetupMixin):
    def configure_dataloaders(self):
        # Get steps for progress tracking
        train_steps = get_train_dataloader_steps(
            self.config.training_parameters,
            train_loader
        )
        
        val_steps = get_val_dataloader_steps(
            self.config.training_parameters,
            val_loader
        )
        
        self._num_train_steps = train_steps
        self._num_val_steps = val_steps
        
        return train_loader, val_loader
```

## Custom Batch Processing

### Collate Functions

```python
def custom_collate_fn(samples: list[dict]) -> Batch:
    # Process list of samples into batch
    inputs = torch.stack([s["input"] for s in samples])
    targets = torch.stack([s["target"] for s in samples])
    
    # Dynamic padding
    max_len = max(len(s["sequence"]) for s in samples)
    padded_sequences = torch.zeros(len(samples), max_len)
    
    for i, s in enumerate(samples):
        seq_len = len(s["sequence"])
        padded_sequences[i, :seq_len] = s["sequence"]
    
    return {
        "input": inputs,
        "target": targets,
        "sequences": padded_sequences,
        "lengths": torch.tensor([len(s["sequence"]) for s in samples])
    }

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=custom_collate_fn
)
```

### Batch Transformations

```python
class BatchTransform:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: list[dict]) -> Batch:
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Tokenize batch
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(labels)
        }
```

## Performance Tips

### 1. Prefetching

```python
# Increase prefetch factor for CPU-bound loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    prefetch_factor=4  # Prefetch 4 batches per worker
)
```

### 2. Avoid Bottlenecks

```python
# Profile data loading
import time

def profile_dataloader(dataloader, num_batches: int = 100):
    start = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Ensure GPU sync
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"Average batch time: {elapsed / num_batches:.3f}s")
    print(f"Throughput: {num_batches / elapsed:.1f} batches/s")
```

### 3. Data Pipeline Optimization

```python
# Chain transforms efficiently
transform_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Use torch.compile for transforms
compiled_transform = torch.compile(transform_pipeline)
```

## See Also

- [DataLoaderSetupMixin](../mixins/setup.md#dataloadersetupmixin) - DataLoader configuration
- [Training Configuration](../configuration/training.md) - Batch size settings
- [Distributed Training](../../parallelism.md) - Multi-GPU data loading 