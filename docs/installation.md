# Installation Guide

This comprehensive guide covers all aspects of installing Dream Trainer, from basic setup to production deployment.

## Requirements

### Python Version Requirements

Dream Trainer requires Python 3.10 or later. Here's the compatibility matrix:

| Dream Trainer Version | Python Versions | PyTorch Versions | CUDA Versions |
|----------------------|-----------------|------------------|---------------|
| 0.1.x                | 3.10 - 3.12     | 2.7.1+          | 11.8, 12.1, 12.4 |

### System Requirements

#### Minimum Requirements
- **CPU**: x86_64 or ARM64 processor
- **RAM**: 8GB (for basic usage)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows (WSL2 recommended)

#### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA capability 7.0+ (V100, A100, H100, RTX 3090+)
- **RAM**: 32GB+ (for multi-GPU training)
- **Storage**: 50GB+ free space (for datasets and checkpoints)
- **Network**: High-speed interconnect for multi-node training (InfiniBand preferred)

### PyTorch and CUDA Compatibility

Dream Trainer builds on PyTorch's distributed training capabilities. Ensure your PyTorch installation matches your CUDA version:

```bash
# Check your CUDA version
nvidia-smi

# Check PyTorch and CUDA compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

#### CUDA Compatibility Table

| CUDA Version | PyTorch Install Command |
|--------------|------------------------|
| CUDA 11.8    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| CUDA 12.1    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 12.4+   | `pip install torch torchvision torchaudio` |
| CPU only     | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |

## Installation Methods

### 1. Basic Installation

The simplest way to install Dream Trainer:

```bash
pip install dream-trainer
```

This installs the core framework with minimal dependencies:
- PyTorch (>= 2.7.1)
- loguru (for logging)
- tqdm (for progress bars)
- dist-util (distributed utilities)

### 2. Feature-Specific Installation

Dream Trainer uses optional dependencies to keep the core lightweight. Install only what you need:

```bash
# Weights & Biases Integration
# Includes: wandb with media logging support
pip install "dream-trainer[wandb]"

# TorchMetrics Integration
# Includes: torchmetrics for standardized metric computation
pip install "dream-trainer[metrics]"

# Quantization Support
# Includes: torchao for FP8/INT8 quantization
pip install "dream-trainer[torchao]"

# Fault Tolerance
# Includes: torchft for automatic failure recovery
pip install "dream-trainer[torchft]"

# Enhanced CLI Output
# Includes: rich for better terminal formatting
pip install "dream-trainer[rich]"

# Multiple features
pip install "dream-trainer[wandb,metrics,torchao]"

# All features
pip install "dream-trainer[all]"
```

### 3. Development Installation

For contributing or customizing Dream Trainer:

```bash
# Clone the repository
git clone https://github.com/dream3d-ai/dream-trainer.git
cd dream-trainer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 4. Docker Installation

For reproducible environments and cluster deployment:

#### Using Pre-built Images

```bash
# Pull the latest image
docker pull dream3d/dream-trainer:latest

# Run with GPU support
docker run --gpus all -it dream3d/dream-trainer:latest

# With specific CUDA version
docker pull dream3d/dream-trainer:cuda12.1-pytorch2.7
```

#### Building Custom Image

Create a `Dockerfile`:

```dockerfile
# Base image with CUDA and PyTorch
FROM pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Dream Trainer
RUN pip install dream-trainer[all]

# Optional: Install your project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python"]
```

Build and run:

```bash
docker build -t my-dream-trainer .
docker run --gpus all -v $(pwd):/workspace -it my-dream-trainer
```

### 5. Cluster-Specific Installations

#### SLURM Clusters

Create a module file or use a shared environment:

```bash
# Load required modules
module load python/3.11
module load cuda/12.1
module load gcc/11.2

# Create shared environment
python -m venv /shared/envs/dream-trainer
source /shared/envs/dream-trainer/bin/activate

# Install with MPI support
pip install dream-trainer[all]
pip install mpi4py  # For MPI-based launchers
```

Example SLURM job script:

```bash
#!/bin/bash
#SBATCH --job-name=dream-trainer
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

# Load environment
source /shared/envs/dream-trainer/bin/activate

# Run training
srun python train.py
```

#### Kubernetes Deployment

Create a Kubernetes manifest:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: dream-trainer-job
spec:
  parallelism: 4
  template:
    spec:
      containers:
      - name: trainer
        image: dream3d/dream-trainer:latest
        resources:
          limits:
            nvidia.com/gpu: 8
        volumeMounts:
        - name: data
          mountPath: /data
        - name: checkpoints
          mountPath: /checkpoints
        env:
        - name: MASTER_ADDR
          value: "dream-trainer-master"
        - name: MASTER_PORT
          value: "29500"
        command: ["python", "train.py"]
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data
      - name: checkpoints
        persistentVolumeClaim:
          claimName: model-checkpoints
```

## Verification

### 1. Basic Verification

Verify your installation is working correctly:

```python
# verify_installation.py
import dream_trainer
import torch
import sys

def check_installation():
    """Comprehensive installation check"""
    print(f"Python version: {sys.version}")
    print(f"Dream Trainer version: {dream_trainer.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check optional features
    features = {
        "wandb": "wandb",
        "torchmetrics": "torchmetrics", 
        "torchao": "torchao",
        "torchft": "torchft",
        "rich": "rich"
    }
    
    print("\nOptional features:")
    for name, module in features.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (install with: pip install dream-trainer[{name}])")
    
    # Test basic functionality
    print("\nTesting basic trainer creation...")
    try:
        from dream_trainer import BaseTrainer, BaseTrainerConfig
        from dream_trainer.trainer.mixins import SetupMixin
        
        class TestTrainer(BaseTrainer, SetupMixin):
            def configure_models(self):
                self.model = torch.nn.Linear(10, 1)
            def configure_optimizers(self):
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            def training_step(self, batch, batch_idx):
                return {"loss": torch.tensor(1.0)}
        
        config = BaseTrainerConfig(training_parameters={"n_epochs": 1})
        print("  ✓ Trainer creation successful")
        
    except Exception as e:
        print(f"  ✗ Trainer creation failed: {e}")
    
    return True

if __name__ == "__main__":
    check_installation()
```

Run the verification script (available in `examples/verify_installation.py`):

```bash
python examples/verify_installation.py
```

### 2. Performance Verification

Test that your GPU setup is performing optimally using the provided benchmark script:

```bash
python examples/benchmark_gpu.py
```

Or use the code below for a custom benchmark:

```python
# benchmark.py
import torch
import time
from dream_trainer.utils import get_rank, get_world_size

def benchmark_gpu():
    """Simple GPU benchmark"""
    if not torch.cuda.is_available():
        print("No GPU available for benchmarking")
        return
    
    device = torch.cuda.current_device()
    print(f"Benchmarking GPU {device}: {torch.cuda.get_device_name(device)}")
    
    # Test different sizes
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        # Create random matrices
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Warmup
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        # Calculate TFLOPS
        flops = 2 * size ** 3 * iterations
        duration = end - start
        tflops = flops / duration / 1e12
        
        print(f"  Matrix size {size}x{size}: {tflops:.2f} TFLOPS")
    
    # Test distributed communication if available
    if get_world_size() > 1:
        print(f"\nDistributed setup detected: rank {get_rank()}/{get_world_size()}")
        tensor = torch.ones(1000, 1000, device='cuda')
        start = time.time()
        torch.distributed.all_reduce(tensor)
        torch.cuda.synchronize()
        duration = time.time() - start
        print(f"  All-reduce latency: {duration*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_gpu()
```

### 3. Multi-GPU Verification

Test distributed training setup:

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=2 verify_distributed.py

# Multiple nodes (replace with your hostnames)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=node1 --master_port=29500 verify_distributed.py
```

```python
# verify_distributed.py
import torch
import torch.distributed as dist
from dream_trainer import DreamTrainer, DreamTrainerConfig

def verify_distributed():
    """Verify distributed training setup"""
    if not dist.is_initialized():
        print("Distributed not initialized!")
        return
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size} initialized successfully")
    
    # Test all-reduce
    tensor = torch.ones(1) * rank
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    
    dist.all_reduce(tensor)
    expected = sum(range(world_size))
    
    if tensor.item() == expected:
        print(f"Rank {rank}: All-reduce test PASSED")
    else:
        print(f"Rank {rank}: All-reduce test FAILED (got {tensor.item()}, expected {expected})")

if __name__ == "__main__":
    # Initialize distributed
    torch.distributed.init_process_group(backend="nccl")
    verify_distributed()
```

## Troubleshooting

### Common Installation Issues

#### 1. CUDA/PyTorch Mismatch

**Problem**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**:
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Missing System Libraries

**Problem**: `ImportError: libcudnn.so.8: cannot open shared object file`

**Solution**:
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libcudnn8

# Or use conda
conda install -c conda-forge cudnn
```

#### 3. Memory Issues During Installation

**Problem**: `MemoryError` or killed process during pip install

**Solution**:
```bash
# Use --no-cache-dir to reduce memory usage
pip install --no-cache-dir dream-trainer

# Or increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Distributed Training Issues

**Problem**: `torch.distributed.DistNetworkError`

**Solution**:
```bash
# Check network connectivity
ping <other-node-hostname>

# Check firewall rules (allow PyTorch's default port)
sudo ufw allow 29500

# Test with different backend
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python train.py  # Will show detailed error messages
```

#### 5. Import Errors After Installation

**Problem**: `ModuleNotFoundError: No module named 'dream_trainer'`

**Solution**:
```bash
# Ensure you're in the correct environment
which python
pip list | grep dream-trainer

# If using conda/venv, activate it
conda activate myenv  # or
source venv/bin/activate

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Performance Optimization Tips

1. **Enable NVIDIA Apex** (if available):
```bash
pip install -v --disable-pip-version-check --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    git+https://github.com/NVIDIA/apex
```

2. **Set optimal environment variables**:
```bash
# Optimize NCCL performance
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
```

3. **Verify GPU-Direct RDMA** (for multi-node):
```bash
# Check if available
nvidia-smi topo -m

# Enable P2P access
sudo nvidia-smi -pm 1
```

## Next Steps

After successful installation:

1. **Follow the Getting Started Guide**: Learn the basics of Dream Trainer
2. **Explore Examples**: Check out the `examples/` directory
3. **Read the API Documentation**: Understand the framework components
4. **Join the Community**: Get help and share experiences

## Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](/faq)
2. Search [GitHub Issues](https://github.com/dream3d-ai/dream-trainer/issues)
3. Join our [Discord Community](https://discord.gg/dream-trainer)
4. Create a [new issue](https://github.com/dream3d-ai/dream-trainer/issues/new) with:
   - Your system information (OS, Python, PyTorch, CUDA versions)
   - Complete error messages
   - Minimal reproduction code 