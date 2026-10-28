"""Persist native torch.compile caches without changing training state."""

import hashlib
import json
import os
import platform
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from dream_trainer.trainer.base import BaseTrainer
from dream_trainer.utils import logger

from .callback import Callback


class CompileCacheCallback(Callback[BaseTrainer]):
    """Reuse compiler caches from a trusted directory across training launches.

    Cache misses still compile normally. These executable artifacts contain no
    training checkpoint and do not guarantee a compilation-free first step.
    Each rank writes atomically to its own file. Share a directory only between
    trusted jobs; use separate directories for experiments that should not
    replace each other's cache bundles.
    """

    def __init__(self, directory: str | Path, save_every_n_steps: int = 100):
        if type(save_every_n_steps) is not int or save_every_n_steps < 1:
            raise ValueError("save_every_n_steps must be a positive integer")
        self.directory = Path(directory)
        self.save_every_n_steps = save_every_n_steps
        self.cache_path: Path | None = None
        self._identity: dict[str, Any] = {}
        self._updates = 0

    def pre_configure(self):
        self.cache_path = None
        self._updates = 0
        if not self.trainer.device_parameters.compile_model:
            return
        device = self.trainer.world.device
        mesh = self.trainer.world.world_mesh
        try:
            triton_version = version("triton")
        except PackageNotFoundError:
            triton_version = None
        self._identity = {
            "format": 1,
            "torch": str(torch.__version__),
            "triton": triton_version,
            "python": platform.python_version(),
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None,
            "mesh": mesh.mesh.tolist() if mesh is not None else None,
            "dimensions": list(mesh.mesh_dim_names or ()) if mesh is not None else [],
            "backend": str(dist.get_backend()) if dist.is_initialized() else "local",
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }
        namespace = hashlib.sha256(
            json.dumps(self._identity, sort_keys=True).encode()
        ).hexdigest()
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.cache_path = self.directory / namespace / f"rank-{rank}.cache"
        try:
            header, payload = self.cache_path.read_bytes().split(b"\n", 1)
            metadata = json.loads(header)
            if metadata["identity"] != self._identity:
                raise ValueError("incompatible compiler cache metadata")
            if metadata["sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError("compiler cache checksum mismatch")
            info = torch.compiler.load_cache_artifacts(payload)
            if info is None:
                raise ValueError("PyTorch rejected the compiler cache")
            logger.info(f"Loaded compiler cache from {self.cache_path}")
        except FileNotFoundError:
            logger.info(f"Compiler cache miss: {self.cache_path}")
        except Exception as error:
            logger.warning(f"Compiler cache miss at {self.cache_path}: {error}")

    def _save(self):
        if self.cache_path is None:
            return
        temporary = None
        try:
            artifacts = torch.compiler.save_cache_artifacts()
            if artifacts is None:
                return
            payload, _ = artifacts
            header = json.dumps(
                {
                    "identity": self._identity,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ).encode()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.cache_path.parent, delete=False) as file:
                temporary = Path(file.name)
                file.write(header + b"\n" + payload)
            os.replace(temporary, self.cache_path)
            logger.info(f"Saved compiler cache to {self.cache_path}")
        except Exception as error:
            logger.warning(f"Could not save compiler cache to {self.cache_path}: {error}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning(
                        f"Could not remove temporary compiler cache {temporary}: {error}"
                    )

    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        if self.cache_path is None or self.trainer.is_accumulating_gradients:
            return
        self._updates += 1
        if self._updates == 1 or self._updates % self.save_every_n_steps == 0:
            self._save()

    def post_fit(self):
        self._save()
