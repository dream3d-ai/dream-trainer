from dataclasses import dataclass, field
from math import prod
from typing import Literal

import dist_util
import torch
import torch.distributed.tensor.parallel

from dream_trainer.utils import logger


@dataclass
class TrainingParameters:
    n_epochs: int | None = None
    train_steps_per_epoch: int | None = None
    train_batch_size: int = 1
    val_steps_per_epoch: int | None = None
    val_frequency: float = 1.0
    num_sanity_val_steps: int = 0
    gradient_clip_val: float | None = None


@dataclass
class Comm:
    init_timeout_seconds: int = 300
    """Timeout for communication operations, during initialization and first train step."""

    train_timeout_seconds: int = 300
    """
    Timeout for communication operations after the first train step --
    usually a tighter bound than during initialization.
    """

    trace_buf_size: int = 20000
    """Flight recorder ring buffer size, >0 means recording by default, 0 means disabled"""


@dataclass(kw_only=True)
class DeviceParameters:
    ## Mixed Precision
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    cpu_offload: bool = False

    ## Model Compilation
    checkpoint_activations: bool = False
    enable_compiled_autograd: bool = False
    compile_model: bool = True
    async_tensor_parallel: bool = True
    loss_parallel: bool = False

    ## Communication
    comm: Comm = field(default_factory=Comm)

    ## Parallelism Dimension Sizes

    # ‣ Weight Sharding (FSDP / HSDP)
    _dp_shard: int | Literal["auto"] = 1
    """
    The number of Data Parallel Shards.
        • >1 → FSDP-style sharding is ON  
        • =1 → sharding OFF

    If you ALSO set `data_parallel_replicate_degree` > 1, you get **HSDP**:
    each replica group uses FSDP internally.
    """

    # ‣ Full Replication (DDP / HSDP)
    _dp_replicate: int | Literal["auto"] = "auto"
    """
    The amount of Data Parallel Replication or (i.e. the number of full *copies* of the model)

    • >1 → DDP-style full replication is ON  
    • =1 → replication OFF

    Combine this with `data_parallel_shard_degree` > 1 for HSDP:
    → *groups* of FSDP-sharded ranks are themselves replicated.
    """

    _tensor_parallel: int | Literal["auto"] = 1
    """
    The amount of Tensor Parallelism.
    """

    _context_parallel: int | Literal["auto"] = 1
    """
    Number of context parallel stages / physical ranks.
    """

    _pipeline_parallel: int | Literal["auto"] = 1
    """
    Number of pipeline stages / physical ranks.

    Still count *physical* ranks even if you use virtual-stage (looped)
    schedules; the scheduler infers virtual stages from the split-points.
    """

    context_parallel_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    """
    Collective used in Context-Parallel SDPA to swap KV shards:

    • "allgather" → pull all shards to every rank after first chunk  
    • "alltoall"  → direct shard-for-shard swap
    """

    ## Debug
    run_single_device_as_fsdp: bool = False
    run_single_device_as_ddp: bool = False

    def __post_init__(self):
        if self.run_single_device_as_fsdp and self.run_single_device_as_ddp:
            raise ValueError("Cannot run single device as both FSDP and DDP")

    @classmethod
    def HSDP(
        cls,
        dp_shard: int,
        tensor_parallel: int | Literal["auto"] = "auto",
        compile_model: bool = True,
        cpu_offload: bool = False,
        checkpoint_activations: bool = False,
        enable_compiled_autograd: bool = False,
    ) -> "DeviceParameters":
        """
        Hybrid Sharding Parallelism
        """

        if tensor_parallel == "auto":
            assert isinstance(dp_shard, int), (
                "dp_shard must be an integer if tensor_parallel is auto"
            )
            tensor_parallel = dist_util.core.get_dist_local_world_size()

        return cls(
            _tensor_parallel=tensor_parallel,
            compile_model=compile_model,
            cpu_offload=cpu_offload,
            checkpoint_activations=checkpoint_activations,
            _dp_replicate="auto",
            _dp_shard=dp_shard,
            _context_parallel=1,
            _pipeline_parallel=1,
            enable_compiled_autograd=enable_compiled_autograd,
        )

    @classmethod
    def FSDP(
        cls,
        tensor_parallel: int | Literal["auto"] = "auto",
        dp_shard: int | Literal["auto"] = "auto",
        compile_model: bool = True,
        cpu_offload: bool = False,
        checkpoint_activations: bool = False,
        enable_compiled_autograd: bool = False,
    ) -> "DeviceParameters":
        """
        TP within nodes
        DP-Shard (FSDP) across nodes
        """
        global_world_size = dist_util.core.get_dist_world_size()
        local_world_size = dist_util.core.get_dist_local_world_size()
        if tensor_parallel == "auto" and dp_shard == "auto":
            logger.info(
                "Using FSDP with both tensor_parallel and dp_shard set to auto, "
                f"setting dp_shard=1 and tensor_parallel={local_world_size}"
            )
            tensor_parallel = local_world_size
            dp_shard = global_world_size // tensor_parallel
        elif tensor_parallel == "auto":
            assert isinstance(dp_shard, int)
            tensor_parallel = global_world_size // dp_shard
            if tensor_parallel > local_world_size:
                raise ValueError(
                    f"tensor_parallel={tensor_parallel} is greater than the local world size, {local_world_size}"
                )
            logger.info(
                f'Got tensor_parallel="auto". Setting tensor_parallel={tensor_parallel}'
            )
        elif dp_shard == "auto":
            assert isinstance(tensor_parallel, int)
            dp_shard = global_world_size // tensor_parallel
            logger.info(f'Got dp_shard="auto". Setting dp_shard={dp_shard}')

        return cls(
            _tensor_parallel=tensor_parallel,
            compile_model=compile_model,
            cpu_offload=cpu_offload,
            checkpoint_activations=checkpoint_activations,
            _dp_shard=dp_shard,
            _dp_replicate=1,
            _context_parallel=1,
            _pipeline_parallel=1,
            enable_compiled_autograd=enable_compiled_autograd,
        )

    @classmethod
    def DDP(
        cls,
        cpu_offload: bool = False,
        checkpoint_activations: bool = False,
        compile_model: bool = True,
        enable_compiled_autograd: bool = False,
    ) -> "DeviceParameters":
        return cls(
            cpu_offload=cpu_offload,
            checkpoint_activations=checkpoint_activations,
            compile_model=compile_model,
            _dp_replicate="auto",
            _dp_shard=1,
            _tensor_parallel=1,
            _context_parallel=1,
            _pipeline_parallel=1,
            enable_compiled_autograd=enable_compiled_autograd,
        )

    @classmethod
    def SINGLE_DEVICE(
        cls,
        compile_model: bool = True,
        cpu_offload: bool = False,
        checkpoint_activations: bool = False,
    ) -> "DeviceParameters":
        return cls(
            compile_model=compile_model,
            cpu_offload=cpu_offload,
            checkpoint_activations=checkpoint_activations,
            _dp_replicate=1,
            _dp_shard=1,
            _tensor_parallel=1,
            _context_parallel=1,
            _pipeline_parallel=1,
            async_tensor_parallel=False,
        )

    def validate(self):
        parallelism_dimensions = {
            "_dp_shard": self._dp_shard,
            "_dp_replicate": self._dp_replicate,
            "_context_parallel": self._context_parallel,
            "_tensor_parallel": self._tensor_parallel,
            "_pipeline_parallel": self._pipeline_parallel,
        }

        auto_dims = [dim for dim, degree in parallelism_dimensions.items() if degree == "auto"]
        assert len(auto_dims) <= 1, "At most one parallelism dimension can be set to 'auto'"
        assert all(
            degree == "auto" or int(degree) > 0 for degree in parallelism_dimensions.values()
        ), "All parallelism dimensions must be set to 'auto' or a positive integer. Got {self}"

        world_size = dist_util.core.get_dist_world_size()
        if len(auto_dims) == 1:
            remainder = prod(int(d) for d in parallelism_dimensions.values() if d != "auto")
            assert world_size % remainder == 0, (
                f"World size, {world_size} must be divisible by the product of the non-auto dimensions {remainder}. Got {self}"
            )
            setattr(self, auto_dims[0], world_size // remainder)
        else:
            assert prod(map(int, parallelism_dimensions.values())) == world_size, (
                f"The product of the parallelism dimensions must equal the world size, {world_size}. Got {self}"
            )

        if self.async_tensor_parallel and not self.compile_model:
            raise ValueError("Async tensor parallelism requires model compilation")

    @property
    def dp_shard(self) -> int:
        return int(self._dp_shard)

    @property
    def dp_replicate(self) -> int:
        return int(self._dp_replicate)

    @property
    def tensor_parallel(self) -> int:
        return int(self._tensor_parallel)

    @property
    def context_parallel(self) -> int:
        return int(self._context_parallel)

    @property
    def pipeline_parallel(self) -> int:
        return int(self._pipeline_parallel)


class FaultToleranceParameters:
    enable: bool = False
    lighthouse_address: str | None = None

    replica_prefix: str | None = None
    """Prefix for the replica ID. If None, the prefix will be set with the experiment name."""

    min_replica_size: int = 1
    """Minimum number of replicas to use for fault tolerance."""

    max_consecutive_failures: int = 3
    """Maximum number of consecutive failures before stopping the training."""
