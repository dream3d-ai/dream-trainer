# Tutorial: Production Setup

This tutorial covers deploying Dream Trainer to production environments, including cluster setup, fault tolerance, monitoring, and best practices for reliable large-scale training.

## What You'll Learn

- Setting up production-grade training infrastructure
- Configuring fault tolerance and checkpointing
- Monitoring and logging strategies
- Optimizing for cluster environments
- Debugging production issues

## Prerequisites

- Completed [Multi-GPU Training](multi-gpu.md)
- Access to a compute cluster (SLURM, Kubernetes, etc.)
- Familiarity with distributed systems concepts

---

## Part 1: Production Configuration

### Robust Configuration Structure

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import (
    SetupMixin, SetupConfigMixin,
    EvalMetricMixin, EvalMetricConfigMixin,
    WandBLoggerMixin, WandBLoggerConfigMixin,
)
from dream_trainer.configs import (
    DeviceParameters,
    TrainingParameters,
    CheckpointParameters,
)
from dream_trainer.callbacks import CallbackCollection


@dataclass
class ProductionConfig(
    DreamTrainerConfig,
    SetupConfigMixin,
    EvalMetricConfigMixin,
    WandBLoggerConfigMixin,
):
    """Production-ready training configuration.

    All paths and settings are configurable via environment
    variables or explicit parameters.
    """
    # === Experiment Tracking ===
    experiment_name: str = "production-run"
    run_id: Optional[str] = None  # Auto-generated if None

    # === Paths (use environment variables for cluster portability) ===
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATA_DIR", "/data"))
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
    )
    log_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("LOG_DIR", "/logs"))
    )

    # === Model Configuration ===
    model_name: str = "llama-7b"
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32

    # === Training Hyperparameters ===
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: Optional[int] = None

    # === Data Configuration ===
    sequence_length: int = 2048
    num_workers: int = 8

    def __post_init__(self):
        """Validate and set up configuration."""
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate run ID if not provided
        if self.run_id is None:
            import datetime
            self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def validate(self):
        """Validate configuration before training."""
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"
        assert self.hidden_size % self.num_heads == 0
        assert self.learning_rate > 0
        assert self.warmup_steps >= 0


def create_production_config(
    num_gpus: int = 8,
    num_nodes: int = 1,
) -> ProductionConfig:
    """Factory function for production configuration."""
    import torch

    return ProductionConfig(
        # Device configuration
        device_parameters=DeviceParameters(
            # FSDP across all GPUs
            dp_shard=num_gpus * num_nodes,
            # Mixed precision
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            # Performance optimizations
            compile_model=True,
            compiled_autograd=True,
            checkpoint_activations=True,
        ),
        # Training configuration
        training_parameters=TrainingParameters(
            n_epochs=1,  # Use max_steps instead for LLMs
            train_batch_size=4,  # Per-GPU
            gradient_accumulation_steps=8,
            gradient_clip_val=1.0,
            val_frequency=0.1,  # Validate 10 times per epoch
            num_sanity_val_steps=2,
        ),
    )
```

### Environment-Based Configuration

```python
# config.py
import os
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Configuration that adapts to deployment environment."""

    @classmethod
    def from_environment(cls) -> "EnvironmentConfig":
        """Create config from environment variables."""
        return cls(
            # Cluster info
            num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
            gpus_per_node=int(os.environ.get("SLURM_GPUS_PER_NODE", 8)),
            node_rank=int(os.environ.get("SLURM_NODEID", 0)),

            # Paths
            data_dir=os.environ.get("DATA_DIR", "/scratch/data"),
            output_dir=os.environ.get("OUTPUT_DIR", "/scratch/output"),

            # W&B
            wandb_project=os.environ.get("WANDB_PROJECT", "production"),
            wandb_entity=os.environ.get("WANDB_ENTITY", None),
        )

    num_nodes: int = 1
    gpus_per_node: int = 8
    node_rank: int = 0
    data_dir: str = "/data"
    output_dir: str = "/output"
    wandb_project: str = "production"
    wandb_entity: str | None = None
```

---

## Part 2: Fault Tolerance

### Checkpoint Strategy

```python
from dream_trainer.callbacks import (
    CheckpointCallback,
    AsyncCheckpointCallback,
    CallbackCollection,
)
from dream_trainer.configs import CheckpointParameters


def create_checkpoint_callbacks(config: ProductionConfig) -> list:
    """Create robust checkpointing strategy."""
    return [
        # Regular checkpoints (synchronous, for safety)
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_dir=config.checkpoint_dir / "regular",
                checkpoint_every_n_steps=1000,
                keep_top_k=5,
                monitor="val_loss",
                mode="min",
            )
        ),

        # Frequent async checkpoints (for fast recovery)
        AsyncCheckpointCallback(
            CheckpointParameters(
                checkpoint_dir=config.checkpoint_dir / "async",
                checkpoint_every_n_steps=100,
                keep_top_k=2,
            )
        ),

        # Epoch checkpoints (for milestone tracking)
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_dir=config.checkpoint_dir / "epochs",
                checkpoint_every_n_epochs=1,
                keep_top_k=None,  # Keep all epoch checkpoints
            )
        ),
    ]
```

### Fault-Tolerant Training with torchft

```python
from dream_trainer.trainer.world import FaultTolerantWorld
from dream_trainer.callbacks import FaultToleranceCallback


class FaultTolerantTrainer(DreamTrainer):
    """Trainer with automatic fault tolerance."""

    def configure(self):
        """Configure with fault-tolerant world."""
        super().configure()

        # Wrap world with fault tolerance
        if self.config.enable_fault_tolerance:
            self.world = FaultTolerantWorld(
                self.world,
                min_replicas=self.config.min_replicas,
                max_restarts=self.config.max_restarts,
            )


# Configuration
@dataclass
class FaultTolerantConfig(ProductionConfig):
    enable_fault_tolerance: bool = True
    min_replicas: int = 4  # Minimum GPUs to continue training
    max_restarts: int = 3  # Max restarts per worker


# Usage
config = FaultTolerantConfig(
    callbacks=CallbackCollection([
        FaultToleranceCallback(
            checkpoint_every_n_steps=50,
            auto_resume=True,
        ),
        *create_checkpoint_callbacks(config),
    ]),
)
```

### Automatic Resume from Checkpoint

```python
from pathlib import Path
import torch.distributed.checkpoint as dcp


def get_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent checkpoint."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def main():
    config = create_production_config()

    # Auto-resume from latest checkpoint
    latest_ckpt = get_latest_checkpoint(config.checkpoint_dir)

    trainer = ProductionTrainer(config)

    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        dcp.load(trainer.state_dict(), checkpoint_id=str(latest_ckpt))

    trainer.fit()
```

---

## Part 3: Monitoring and Logging

### Comprehensive Logging Setup

```python
from dream_trainer.callbacks import (
    LoggerCallback,
    MetricLoggerCallback,
    ModelWatchCallback,
    LRLoggerCallback,
    ProfileCallback,
    TrainerSummary,
)


def create_monitoring_callbacks(config: ProductionConfig) -> list:
    """Create comprehensive monitoring setup."""
    return [
        # Training metrics
        LoggerCallback(
            log_every_n_train_batches=10,
            log_every_n_val_batches=50,
        ),

        # Evaluation metrics (if using EvalMetricMixin)
        MetricLoggerCallback(),

        # Learning rate tracking
        LRLoggerCallback(log_every_n_steps=10),

        # Model architecture logging (once at start)
        ModelWatchCallback(
            log_freq=1000,
            log_gradients=True,
            log_parameters=True,
        ),

        # Performance profiling (periodic)
        ProfileCallback(
            profile_steps=range(100, 110),  # Profile steps 100-109
            output_dir=config.log_dir / "profiles",
            activities=["cpu", "cuda"],
            with_stack=True,
        ),

        # Summary statistics
        TrainerSummary(print_every_n_epochs=1),
    ]
```

### Custom Production Metrics

```python
import time
from dream_trainer.callbacks import Callback


class ProductionMetricsCallback(Callback):
    """Track production-relevant metrics."""

    def __init__(self):
        super().__init__()
        self._step_times = []
        self._epoch_start_time = None

    def pre_train_epoch(self):
        self._epoch_start_time = time.time()
        self._step_times = []

    def post_train_step(self, output, batch_idx):
        step_time = time.time()
        if self._step_times:
            self._step_times.append(step_time - self._step_times[-1])

        # Log throughput metrics
        if batch_idx % 100 == 0 and self._step_times:
            avg_step_time = sum(self._step_times[-100:]) / len(self._step_times[-100:])
            samples_per_sec = (
                self.trainer.config.training_parameters.train_batch_size
                * self.trainer.world.world_size
                / avg_step_time
            )

            if hasattr(self.trainer, 'log_scalar'):
                self.trainer.log_scalar("perf/samples_per_sec", samples_per_sec)
                self.trainer.log_scalar("perf/step_time_ms", avg_step_time * 1000)

    def post_train_epoch(self):
        if self._epoch_start_time:
            epoch_time = time.time() - self._epoch_start_time
            if hasattr(self.trainer, 'log_scalar'):
                self.trainer.log_scalar("perf/epoch_time_sec", epoch_time)
```

### W&B Integration

```python
from dream_trainer.trainer.mixins import WandBLoggerMixin, WandBLoggerConfigMixin


@dataclass
class WandBProductionConfig(ProductionConfig, WandBLoggerConfigMixin):
    # W&B settings
    wandb_project: str = "production-training"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["production"])
    wandb_notes: str = ""

    # Artifact logging
    log_model_artifacts: bool = True
    artifact_save_frequency: int = 5000  # steps


class WandBProductionTrainer(DreamTrainer, WandBLoggerMixin):
    """Production trainer with W&B integration."""

    config: WandBProductionConfig

    def post_setup(self):
        """Log experiment configuration to W&B."""
        super().post_setup()

        if self.world.rank == 0:
            import wandb

            # Log config
            wandb.config.update({
                "model_name": self.config.model_name,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "learning_rate": self.config.learning_rate,
                "world_size": self.world.world_size,
            })

            # Log model summary
            wandb.watch(
                self.model,
                log="gradients",
                log_freq=1000,
            )
```

---

## Part 4: Cluster Deployment

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=dream-trainer-prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# === Environment Setup ===
module load cuda/12.1
module load python/3.11

source /path/to/venv/bin/activate

# === Environment Variables ===
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Data and output paths
export DATA_DIR=/scratch/datasets/my_dataset
export CHECKPOINT_DIR=/scratch/checkpoints/$SLURM_JOB_ID
export LOG_DIR=/scratch/logs/$SLURM_JOB_ID

# W&B configuration
export WANDB_PROJECT=production-training
export WANDB_RUN_ID=$SLURM_JOB_ID

# NCCL optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# === Create directories ===
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# === Launch Training ===
srun --kill-on-bad-exit=1 torchrun \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config production \
    --experiment-name "prod-$SLURM_JOB_ID"
```

### Kubernetes Deployment

```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: dream-trainer-production
  labels:
    app: dream-trainer
spec:
  parallelism: 4  # Number of nodes
  completions: 4
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: dream-trainer
    spec:
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: dream3d/dream-trainer:production
        resources:
          requests:
            memory: "400Gi"
            cpu: "64"
            nvidia.com/gpu: 8
          limits:
            memory: "500Gi"
            cpu: "64"
            nvidia.com/gpu: 8
        env:
        - name: MASTER_ADDR
          value: "dream-trainer-production-0.dream-trainer-headless"
        - name: MASTER_PORT
          value: "29500"
        - name: WORLD_SIZE
          value: "32"  # 4 nodes * 8 GPUs
        - name: DATA_DIR
          value: "/data"
        - name: CHECKPOINT_DIR
          value: "/checkpoints"
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-secret
              key: api-key
        volumeMounts:
        - name: data
          mountPath: /data
          readOnly: true
        - name: checkpoints
          mountPath: /checkpoints
        - name: shm
          mountPath: /dev/shm
        command:
        - torchrun
        - --nproc_per_node=8
        - --nnodes=4
        - --node_rank=$(JOB_COMPLETION_INDEX)
        - --master_addr=$(MASTER_ADDR)
        - --master_port=$(MASTER_PORT)
        - train.py
        - --config=production
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: checkpoints
        persistentVolumeClaim:
          claimName: checkpoints-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 64Gi

---
# Headless service for DNS resolution
apiVersion: v1
kind: Service
metadata:
  name: dream-trainer-headless
spec:
  clusterIP: None
  selector:
    app: dream-trainer
  ports:
  - port: 29500
    name: nccl
```

### Docker Image

```dockerfile
# Dockerfile.production
FROM pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Dream Trainer
RUN pip install --no-cache-dir dream-trainer[all]

# Copy training code
COPY . /app
WORKDIR /app

# Set up entrypoint
ENTRYPOINT ["python", "train.py"]
```

---

## Part 5: Debugging Production Issues

### Debug Mode Configuration

```python
@dataclass
class DebugConfig(ProductionConfig):
    """Configuration for debugging production issues."""
    debug_mode: bool = True
    debug_samples: int = 100
    profile_memory: bool = True
    trace_distributed: bool = True


def enable_debug_mode(config: ProductionConfig):
    """Enable comprehensive debugging."""
    import os

    # Distributed debugging
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

    # PyTorch debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Memory debugging
    import torch
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
    )
```

### Memory Debugging Callback

```python
import torch
from dream_trainer.callbacks import RankZeroCallback


class MemoryDebugCallback(RankZeroCallback):
    """Debug memory usage during training."""

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def post_train_step(self, output, batch_idx):
        if batch_idx % self.log_every_n_steps != 0:
            return

        if not torch.cuda.is_available():
            return

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        print(f"Step {batch_idx} Memory (GB):")
        print(f"  Allocated: {allocated:.2f}")
        print(f"  Reserved: {reserved:.2f}")
        print(f"  Max Allocated: {max_allocated:.2f}")

        # Log to W&B if available
        if hasattr(self.trainer, 'log_scalar'):
            self.trainer.log_scalar("memory/allocated_gb", allocated)
            self.trainer.log_scalar("memory/reserved_gb", reserved)
            self.trainer.log_scalar("memory/max_allocated_gb", max_allocated)

    def on_interrupt(self, exception):
        """Dump memory snapshot on error."""
        if torch.cuda.is_available():
            torch.cuda.memory._dump_snapshot(f"memory_snapshot_{self.trainer.global_step}.pickle")
```

### Distributed Debugging

```python
import torch.distributed as dist


class DistributedDebugCallback(Callback):
    """Debug distributed training issues."""

    def post_setup(self):
        """Log distributed configuration."""
        rank = self.trainer.world.rank
        world_size = self.trainer.world.world_size

        print(f"[Rank {rank}/{world_size}] Setup complete")
        print(f"[Rank {rank}] Device: {self.trainer.device}")
        print(f"[Rank {rank}] Model params: {sum(p.numel() for p in self.trainer.model.parameters()):,}")

        # Verify all ranks have same config
        config_hash = hash(str(self.trainer.config))
        all_hashes = [None] * world_size
        dist.all_gather_object(all_hashes, config_hash)

        if rank == 0:
            if len(set(all_hashes)) != 1:
                print("WARNING: Config mismatch across ranks!")
            else:
                print("Config verified across all ranks")

    def pre_train_step(self, batch_idx):
        """Verify data distribution."""
        if batch_idx == 0:
            rank = self.trainer.world.rank
            # Log first batch info
            print(f"[Rank {rank}] First batch loaded")
```

---

## Part 6: Complete Production Example

```python
"""Production training script."""
from dataclasses import dataclass, field
from pathlib import Path
import os
import argparse

import torch
from torch.utils.data import DataLoader

from dream_trainer import DreamTrainer, DreamTrainerConfig
from dream_trainer.trainer.mixins import (
    SetupMixin, SetupConfigMixin,
    EvalMetricMixin, EvalMetricConfigMixin,
    WandBLoggerMixin, WandBLoggerConfigMixin,
)
from dream_trainer.configs import (
    DeviceParameters,
    TrainingParameters,
    CheckpointParameters,
)
from dream_trainer.callbacks import (
    CheckpointCallback,
    AsyncCheckpointCallback,
    LoggerCallback,
    ProgressBar,
    OptimizeFSDP,
    GarbageCollectionCallback,
    CallbackCollection,
)


@dataclass
class ProductionConfig(
    DreamTrainerConfig,
    SetupConfigMixin,
    EvalMetricConfigMixin,
    WandBLoggerConfigMixin,
):
    # Paths
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATA_DIR", "/data"))
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
    )

    # Model
    hidden_size: int = 4096
    num_layers: int = 32

    # Training
    learning_rate: float = 3e-4

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class ProductionTrainer(DreamTrainer, WandBLoggerMixin):
    config: ProductionConfig

    def configure_models(self):
        from my_models import LargeLanguageModel
        self.model = LargeLanguageModel(self.config)

    # ... implement other required methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="production")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Create callbacks
    callbacks = CallbackCollection([
        ProgressBar(),
        LoggerCallback(log_every_n_train_batches=10),
        OptimizeFSDP(prefetch=2),
        GarbageCollectionCallback(gc_every_n_steps=100),
        CheckpointCallback(
            CheckpointParameters(
                checkpoint_dir=Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints")),
                checkpoint_every_n_steps=1000,
                keep_top_k=5,
            )
        ),
        AsyncCheckpointCallback(
            CheckpointParameters(
                checkpoint_dir=Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints")) / "async",
                checkpoint_every_n_steps=100,
                keep_top_k=2,
            )
        ),
    ])

    # Create config
    config = ProductionConfig(
        device_parameters=DeviceParameters(
            dp_shard=int(os.environ.get("WORLD_SIZE", 8)),
            param_dtype=torch.bfloat16,
            compile_model=True,
            checkpoint_activations=True,
        ),
        training_parameters=TrainingParameters(
            n_epochs=1,
            train_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_clip_val=1.0,
        ),
        callbacks=callbacks,
        # W&B
        wandb_project=os.environ.get("WANDB_PROJECT", "production"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )

    # Create and run trainer
    trainer = ProductionTrainer(config)

    # Resume if specified
    if args.resume:
        import torch.distributed.checkpoint as dcp
        dcp.load(trainer.state_dict(), checkpoint_id=args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()
```

---

## Summary

You've learned how to:

1. **Configure** production-grade training with environment variables
2. **Implement fault tolerance** with checkpointing and torchft
3. **Monitor training** with comprehensive logging
4. **Deploy to clusters** using SLURM and Kubernetes
5. **Debug issues** with specialized callbacks

## Next Steps

- [Performance Guide](../performance.md): Optimize training throughput
- [Debugging Guide](../debugging.md): Deep dive into debugging
- [Parallelism Guide](../parallelism.md): Advanced parallelism strategies
