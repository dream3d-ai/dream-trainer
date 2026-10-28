"""Measure the workspace's real Slowrun model, Muon optimizer, and FineWeb tokens.

Launch with torchrun. The token file is an ordered production FineWeb loader
capture (see experiments/slowrun-gst-v2/trainer_benchmark.py cache). A reduced
model makes this a trainer-overhead benchmark, not a full-training quality test.
Use a fresh TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR for every launch.
"""

import hashlib
import faulthandler
import json
import os
import time
from pathlib import Path

import torch
import typer
from torch._dynamo.utils import counters
from torch.utils.data import DataLoader

from dream_trainer.callbacks import Callback, CallbackCollection
from dream_trainer.configs import DeviceParameters
from dream_trainer.trainer.base import BaseTrainer
from models.models.slowrun.baseline.model import GPTConfig
from models.models.slowrun.baseline.train import SlowrunTrainer, SlowrunTrainerConfig


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for member in item:
                visit(member)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def main(
    tokens: Path,
    output: Path,
    backend: str = "nccl",
    cache: Path | None = None,
    steps: int = 20,
    warmup: int = 4,
    compile_model: bool = True,
):
    faulthandler.dump_traceback_later(180, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Hold GEMM implementations fixed across independent cold-cache launches.
    # The production model's max-autotune compile hook remains in use.
    torch._inductor.config.max_autotune_gemm_backends = "ATEN"
    data = torch.load(tokens, map_location="cpu", weights_only=True)
    if data.get("split") != "train" or not data.get("source", "").startswith("hf://"):
        raise ValueError("Expected a production FineWeb training-token capture")
    config = SlowrunTrainerConfig.for_test(
        model=GPTConfig(
            n_layer=4,
            n_head=2,
            n_embd=128,
            sequence_len=128,
            dropout=0.0,
            window_pattern="L",
        )
    )
    config.seed = 123
    config.optimizer.foreach = True
    config.logging_parameters.enabled = False
    config.device_parameters = DeviceParameters.DDP(compile_model=compile_model)
    config.device_parameters.async_tensor_parallel = False
    if hasattr(config.device_parameters.comm, "backend"):
        config.device_parameters.comm.backend = backend
    elif backend != "nccl":
        raise ValueError("Baseline only supports NCCL")
    config.training_parameters.train_steps_per_epoch = steps
    config.training_parameters.val_steps_per_epoch = 0
    config.training_parameters.gradient_accumulation_steps = 1
    config.training_parameters.val_every_n_steps = steps + 1
    report = {
        "backend": backend,
        "compile": compile_model,
        "token_source": data["source"],
        "token_capture_sha256": data["sha256"],
        "warmup": warmup,
        "steps": steps,
        "model": {"layers": 4, "width": 128, "sequence_length": 128, "window_pattern": "L"},
        "gemm_backends": "ATEN",
        "optimizer_foreach": True,
        "deterministic": True,
    }
    samples, losses = [], []
    launched = time.perf_counter()

    class Measure(Callback[BaseTrainer]):
        def pre_optimizer_step(self, model, optimizer):
            if self.trainer.global_step == 0:
                print("First forward/backward and validation complete", flush=True)

        def pre_fit(self):
            report["initial"] = fingerprint(self.trainer.model.state_dict())
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            report["setup_seconds"] = time.perf_counter() - launched

        def pre_train_step(self, batch, batch_idx):
            self.profile = None
            if batch_idx == steps - 1:
                self.profile = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                )
                self.profile.__enter__()
            self.started = time.perf_counter()

        def post_train_step(self, result, batch_idx):
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - self.started)
            losses.append(float(result["train/loss"]))
            if batch_idx == 0:
                report["first_update_seconds"] = time.perf_counter() - launched
                print(
                    f"First update complete: {report['first_update_seconds']:.3f}s", flush=True
                )
            if self.profile is not None:
                self.profile.__exit__(None, None, None)
                report["host_scalar_reads"] = sum(
                    event.count
                    for event in self.profile.key_averages()
                    if event.key == "aten::_local_scalar_dense"
                )

        def post_fit(self):
            trainer = self.trainer
            report.update(
                {
                    "samples_seconds": samples,
                    "losses": losses,
                    "measured_seconds": sum(samples[warmup:-1]),
                    "tokens_per_second": (steps - warmup - 1)
                    * 2
                    * 128
                    / sum(samples[warmup:-1]),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "final": fingerprint(trainer.model.state_dict()),
                    "optimizer": fingerprint(trainer.optimizer.state_dict()),
                    "scheduler": fingerprint(trainer.scheduler.state_dict()),
                    "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(),
                    "compiler_counters": {
                        name: dict(values) for name, values in counters.items()
                    },
                }
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.with_name(
                f"{output.stem}-rank-{trainer.world.world_mesh.get_rank()}.json"
            ).write_text(json.dumps(report, indent=2))

    callbacks = []
    if cache is not None:
        from dream_trainer.callbacks import CompileCacheCallback

        # Keep export outside the steady-state window when comparing validation
        # and communication overhead; still include it in first-update latency.
        callbacks.append(CompileCacheCallback(cache, save_every_n_steps=steps + 1))
    callbacks.append(Measure())
    config.callbacks = CallbackCollection(callbacks)

    class MeasuredTrainer(SlowrunTrainer):
        def configure_dataloaders(self):
            rank, size = self.world.dp_rank, self.world.dp_size
            batches = [
                {key: data[key][index, :128].contiguous() for key in ("input_ids", "targets")}
                for index in range(rank, steps * size, size)
            ]
            report["inputs"] = fingerprint(batches)
            return DataLoader(batches, batch_size=1, pin_memory=True), DataLoader([])

    trainer = MeasuredTrainer(config)
    try:
        trainer.fit()
    finally:
        faulthandler.cancel_dump_traceback_later()
        trainer.finish()
        trainer.world.destroy()


if __name__ == "__main__":
    typer.run(main)
