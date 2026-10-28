# Native trainer upgrade measurements

Recorded results and acceptance evidence:
[2026-09-24 native upgrade](results/2026-09-24-native-upgrade/README.md).

`native_upgrade.py` requires the Dream3D workspace's `models` package. It runs
the existing Slowrun trainer, model setup/compile hook, and Muon optimizer. Input
is a saved, ordered production FineWeb loader capture, not random token data.
The screening model has four layers, width 128, 128-token sequences, the full
GPT-2 vocabulary, full causal attention in every layer, and dropout disabled.
Each of two DDP ranks consumes one
different sequence per update. This measures trainer overhead; it does not
establish full-model throughput, convergence, or training quality.

The benchmark enables deterministic algorithms and holds GEMM selection at ATen
across independent compilations while
retaining the model's `max-autotune-no-cudagraphs` compile hook and the production
optimizer's foreach mode. Full causal attention avoids varying FlexAttention
autotuning choices between controls. Every launch must
use fresh Inductor and Triton directories. Only the explicit compiler bundle is
reused in a cache-load arm. Runs use the same seed, inputs and optimizer settings;
JSON reports contain initial/final model, optimizer, scheduler and input hashes.

From the Dream3D workspace root:

```bash
# Capture real training tokens once, using the production loader.
.venv/bin/python experiments/slowrun-gst-v2/trainer_benchmark.py cache /tmp/fineweb.pt

# Cold build: use new cache directories and an initially empty bundle directory.
OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8 \
TORCHINDUCTOR_CACHE_DIR=/tmp/native-build-inductor \
TRITON_CACHE_DIR=/tmp/native-build-triton \
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  dream-trainer/benchmarks/native_upgrade.py /tmp/fineweb.pt /tmp/native-build.json \
  --backend nccl --cache /tmp/native-bundle

# Restart: empty local caches, same bundle.
OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8 \
TORCHINDUCTOR_CACHE_DIR=/tmp/native-load-inductor \
TRITON_CACHE_DIR=/tmp/native-load-triton \
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  dream-trainer/benchmarks/native_upgrade.py /tmp/fineweb.pt /tmp/native-load.json \
  --backend nccl --cache /tmp/native-bundle
```

To isolate trainer and backend changes, also run without `--cache`, first against
an archived baseline package selected with `PYTHONPATH`, then the updated package
with `--backend nccl` and `--backend nccl2`. Repeat each arm three times in fresh
processes. GPU jobs run sequentially. Compare matching ranks' input and training
state before interpreting timings.

Reports separate setup and time to first completed update from steady-state
throughput. The first four updates are warmup; the last update is profiled for
host scalar reads and excluded from throughput. The benchmark sets the cache-save
interval beyond its update count, so exports happen on the first update and at
shutdown. First-update latency includes export; steady-state comparisons exclude
export. Timing synchronizes at update boundaries and peak
memory is measured after setup. Aggregate distributed throughput using the slower
rank, then report the median across repetitions.

Acceptance fixtures, from the workspace root:

```bash
.venv/bin/python -m pytest dream-trainer/tests -q
OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 dream-trainer/tests/distributed_native_upgrade.py \
  /tmp/native-acceptance --backend nccl2 --mode fsdp --bf16
# Repeat for nccl/nccl2 and ddp/fsdp/tp; BF16 is an additional FSDP case.
```

The distributed fixture checks world/mesh backends, exact checkpoint continuation,
optimizer state, and coordinated rejection of invalid parameters, gradients,
missing gradients and overflowing norms. Unit tests also cover accumulation,
unused/frozen parameters, independent optimizers, callback ordering, cache-write
failure, incompatible artifacts, disabled compilation, and fresh-process cache
reuse after changes to shapes or code constants.
