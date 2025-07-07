import contextlib
import datetime as dt
import math
import os
from typing import Callable, Iterable, cast

import dist_util
import torch
import torch._inductor.config
import torch.distributed
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import ProcessGroup, ReduceOp
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from dream_trainer.configs import DeviceParameters
from dream_trainer.utils import logger
from dream_trainer.utils.common import get_device_info, stacked_context


def construct_mesh(
    config: DeviceParameters,
    device_type: str,
    init_device_mesh_fn: Callable[..., DeviceMesh],
) -> DeviceMesh:
    # NOTE: Pytorch only supports accessing contiguous subspaces of the mesh.
    # FSDP expects TP to be the final dimension (presumably why torchtitan shards over dp_shard+cp)
    # PP/CP first is no bueno (corresponds to physically farther devices)
    dims = [
        config.pipeline_parallel,
        config.dp_replicate,
        config.dp_shard,
        config.context_parallel,
        config.tensor_parallel,
    ]
    names = [
        "pp",
        "dp_replicate",
        "dp_shard",
        "cp",
        "tp",
    ]
    names_and_dims = list(zip(names, dims))
    names_and_dims = [(name, dim) for name, dim in names_and_dims if dim > 1]

    # If running single device as FSDP (or DDP), we add dummy dimensions
    if len(names_and_dims) == 0:
        if config.run_single_device_as_fsdp:
            names_and_dims = [("dp_shard", 1)]
            logger.info("Running single device as FSDP")
        elif config.run_single_device_as_ddp:
            names_and_dims = [("dp_replicate", 1)]
            logger.info("Running single device as DDP")

    filtered_names = [name for name, _ in names_and_dims]
    filtered_dims = [dim for _, dim in names_and_dims]

    mesh = init_device_mesh_fn(device_type, filtered_dims, mesh_dim_names=filtered_names)
    logger.info(
        f"Built {mesh.ndim}D device mesh with {', '.join(f'{name.upper()}={dim}' for name, dim in names_and_dims)}"
    )

    # Helper function to select only existing mesh dimensions
    filter_dims = lambda dims: tuple(dim for dim in dims if dim in filtered_names)

    # Create useful flattened views

    # Data Parallel Dim
    if dims := filter_dims(("dp_replicate", "dp_shard")):
        mesh[dims]._flatten(mesh_dim_name="dp")

    # Fully Sharded Data Parallel Dim
    if dims := filter_dims(("dp_shard", "cp")):
        mesh[dims]._flatten(mesh_dim_name="dp_shard+cp")

    # Context Data parallel dim
    if dims := filter_dims(("dp_replicate", "dp_shard", "cp")):
        mesh[dims]._flatten(mesh_dim_name="dp+cp")

    # Tensor Parallel Dim
    if dims := filter_dims(("cp", "tp")):
        mesh[dims]._flatten(mesh_dim_name="cp+tp")

    return mesh


class DistributedWorld:
    def __init__(self, config: DeviceParameters):
        config.validate()
        self.config = config

        self.world_mesh: DeviceMesh | None = None
        self.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

        # NOTE: `device_module.set_device` has to be set before creating TorchFT manager.
        self.device_type, self.device_module = get_device_info()
        self.device = torch.device(f"{self.device_type}:{int(os.environ['LOCAL_RANK'])}")
        self.device_module.set_device(self.device)

    ##############
    # Properties #
    ##############

    @property
    def param_dtype(self):
        return self.config.param_dtype

    @property
    def reduce_dtype(self):
        return self.config.reduce_dtype

    @property
    def dp_replicate_enabled(self):
        return self.get_mesh("dp_replicate") is not None

    @property
    def dp_shard_enabled(self):
        return self.get_mesh("dp_shard") is not None

    @property
    def tp_enabled(self):
        return self.get_mesh("tp") is not None

    @property
    def pp_enabled(self):
        return self.get_mesh("pp") is not None

    @property
    def cp_enabled(self):
        return self.get_mesh("cp") is not None

    @property
    def loss_parallel_enabled(self):
        return self.config.loss_parallel and self.get_mesh("tp") is not None

    ##########
    # Meshes #
    ##########

    def get_mesh(self, mesh_dim_name: str) -> DeviceMesh | None:
        if self.world_mesh is None:
            raise RuntimeError("World mesh not yet initialized. Call `launch` first.")

        if not self.world_mesh.mesh_dim_names:
            return None

        try:
            return self.world_mesh[mesh_dim_name]
        except KeyError:
            return None

    @property
    def is_global_zero(self):
        return dist_util.core.is_rank_zero()

    @property
    def dp_size(self) -> int:
        dp_mesh = self.get_mesh("dp")
        return dp_mesh.size() if dp_mesh is not None else 1

    @property
    def dp_rank(self) -> int:
        dp_mesh = self.get_mesh("dp")
        return dp_mesh.get_local_rank() if dp_mesh is not None else 0

    @property
    def fsdp_config(self):
        """
        Returns the configuration dictionary for Fully Sharded Data Parallel (FSDP).

        The configuration includes:
            - 'mesh': The data parallel device mesh.
            - 'mp_policy': The mixed precision policy for FSDP, specifying parameter and reduction dtypes.
            - 'offload_policy': (Optional) CPU offload policy if enabled in the config.
            - 'reshard_after_forward': (Optional) Whether to reshard after forward. Only set for PP.

        Returns:
            dict: FSDP configuration dictionary.
        """
        if self.world_mesh is None:
            raise RuntimeError("World mesh not yet initialized. Call `launch` first.")

        if self.get_mesh("dp_shard") is None:
            return None

        if self.get_mesh("dp_replicate") is not None:  # Apply HSDP
            mesh = self.world_mesh[("dp_replicate", "dp_shard+cp")]
        else:
            mesh = self.world_mesh["dp_shard+cp"]

        config = {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=self.param_dtype, reduce_dtype=self.reduce_dtype
            ),
        }
        if self.config.cpu_offload:
            config["offload_policy"] = CPUOffloadPolicy()
        return config

    #########
    # Setup #
    #########

    def launch(self):
        """
        Initializes the distributed process group and sets up the device mesh for parallel training.

        This method performs the following steps:
            1. Initializes the distributed process group with the appropriate backend and timeout.
            2. Builds the world mesh using the parallel dimensions and device type.
            3. Updates data parallel (DP) mesh information if DP is enabled.

        Raises:
            Any exceptions raised by torch.distributed.init_process_group or mesh building.

        Side Effects:
            - Sets self.world_mesh to the constructed mesh.
        """
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = str(self.config.comm.trace_buf_size)

        # Set device-specific backends
        device_backend = c10d.Backend.default_device_backend_map.get(self.device_type)
        cpu_backend = c10d.Backend.default_device_backend_map.get("cpu")
        backend = f"{self.device_type}:{device_backend},cpu:{cpu_backend}"

        # Initialize the process group & build the world mesh
        dist.init_process_group(
            backend=backend,
            timeout=dt.timedelta(seconds=self.config.comm.init_timeout_seconds),
        )
        logger.info(
            f"Initialized Distributed World [Rank {dist.get_rank()}] [Size: {dist.get_world_size()}]"
        )
        self.world_mesh = construct_mesh(self.config, self.device_type, init_device_mesh)

        self.set_pg_timeouts(dt.timedelta(seconds=self.config.comm.init_timeout_seconds))

        if (tp_mesh := self.get_mesh("tp")) is not None:
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            if self.config.async_tensor_parallel:
                # Enable symmetric memory for the TP process group
                enable_symm_mem_for_group(tp_mesh.get_group().group_name)
                # Tell torch.compile to enable async-TP
                torch._inductor.config._micro_pipeline_tp = True
                logger.info("Enabled Async Tensor Parallelism")

        logger.info("Launched Distributed World")

    def set_determinism(
        self,
        seed: int | None = None,
        deterministic: bool = False,
        distinct_seed_mesh_dim: str = "pp",
    ) -> None:
        """
        Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds
        across dimension denoted by `distinct_seed_mesh_dim`. An example use case is pipeline parallelism,
        where we want to have the same seed across SPMD groups, but different seeds across PP groups.

        Currently, does not set seeds for the CUDA RNG since trainer always uses DTensor for SPMD parallelisms,
        and DTensor manages its own RNG tracker, but we could extend to support both if needed.

        Set Determinism flags for increased reproducibility with loss of performance.
        """
        if self.world_mesh is None:
            raise RuntimeError("World mesh not yet initialized. Call `launch` first.")

        if deterministic:
            logger.info("Deterministic algorithm enabled (expect perf degradation).")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # env var for deterministic CuBLAS
            # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.info(f"Single-process job using seed: {seed}")
            return

        # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
        # if user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
        if seed is None:
            # Extract the seed for torch's main generator on rank 0 and standardizes on using that to build
            # seeds for unique SPMD groups
            seed_tensor = torch.get_rng_state()[:8].to(self.device)
            dist.broadcast(seed_tensor, src=0)
            seed = cast(int, seed_tensor.to("cpu").view(torch.uint64).item())

        # Set distinct seed for each rank in mesh dimensions, with dimension name provdied by `distinct_seed_mesh_dim`
        # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
        # and choose a unique seed for each rank on the PP mesh.
        # TODO(jianiw): We could further extend this to support mutiple distinct dimensions instead of just one.
        if (
            c10d.get_world_size() > 1
            and self.world_mesh.mesh_dim_names is not None
            and distinct_seed_mesh_dim in self.world_mesh.mesh_dim_names
        ):
            distinct_mesh = self.world_mesh[distinct_seed_mesh_dim]
            seed += distinct_mesh.get_local_rank()
            seed %= 2**64

            logger.info(
                f"{distinct_seed_mesh_dim} rank {distinct_mesh.get_local_rank()}, Global rank {c10d.get_rank()} using seed: {seed}"
            )
            duplicate_seed_mesh = list(
                filter(
                    lambda name: name != distinct_seed_mesh_dim,
                    self.world_mesh.mesh_dim_names,
                )
            )
            duplicate_seed_mesh = (
                self.world_mesh[tuple(duplicate_seed_mesh)]
                if len(duplicate_seed_mesh)
                else None
            )
        else:
            duplicate_seed_mesh = self.world_mesh
            logger.info(f"Global Rank {c10d.get_rank()} using seed: {seed}")

        # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
        torch.manual_seed(seed)
        # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
        os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

        # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for all ranks of the SPMD mesh.
        # IF PP is also used, this seed is unique per PP rank.
        if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
            torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)

    ####################
    # Context Managers #
    ####################

    @contextlib.contextmanager
    def context_parallel_context(
        self,
        cp_buffers: list[torch.Tensor],
        cp_seq_dims: list[int],
        cp_no_restore_buffers: set[torch.Tensor],
    ):
        """
        Creates and returns a context manager for context parallelism using PyTorch's experimental Context Parallel API.

        Args:
            cp_buffers (list[torch.Tensor]): List of buffers to be used in context parallelism.
            cp_seq_dims (list[int]): List of sequence dimensions for each buffer.
            cp_no_restore_buffers (set[torch.Tensor]): Set of buffers that should not be restored.
            cp_rotate_method (str): The rotation method to be set for context parallelism.

        Returns:
            contextlib._GeneratorContextManager: A context manager for context parallelism.

        Raises:
            ImportError: If the required experimental Context Parallel API is not available in the current PyTorch version.
        """
        try:
            from torch.distributed.tensor.experimental import context_parallel
            from torch.distributed.tensor.experimental._attention import (
                set_rotate_method,
            )
        except ImportError:
            print(
                f"PyTorch version {torch.__version__} does not include the experimental "
                "Context Parallel API. Please update to a newer version."
            )

        if self.world_mesh is None:
            raise RuntimeError("World mesh not yet initialized. Call `launch` first.")

        set_rotate_method(self.config.context_parallel_rotate_method)
        with context_parallel(
            self.world_mesh["cp"],
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=cp_no_restore_buffers,
        ):
            yield

    @contextlib.contextmanager
    def loss_parallel(self):
        """
        Context manager for loss parallelism.

        This context manager enables loss parallelism by setting up the appropriate distributed environment,
        but does nothing if tensor parallelism (tp) is not enabled.

        Yields:
            contextlib._GeneratorContextManager: A context manager that sets up the loss parallelism.
        """
        with (
            torch.distributed.tensor.parallel.loss_parallel()
            if self.loss_parallel_enabled
            else contextlib.nullcontext()
        ):
            yield

    @contextlib.contextmanager
    def train_context(self):
        """
        Returns a context manager for training that sets up optional distributed context features.

        This context manager enables:
            - Compiled autograd if configured (`self.config.enable_compiled_autograd`)

        Returns:
            contextlib._GeneratorContextManager: A context manager that sets up the training context.
        """

        contexts = []
        if self.config.enable_compiled_autograd:
            contexts.append(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

        with stacked_context(contexts):
            yield

    ###################
    # Dist Collective #
    ###################

    def barrier(self, process_group: ProcessGroup | None = None):
        """
        Synchronizes all processes in the specified process group.

        This method blocks until all processes in the group reach this barrier.
        If no process group is specified, the world mesh group is used by default.

        Args:
            process_group (ProcessGroup | None, optional): The process group to synchronize.
                Defaults to None, in which case the world mesh group is used.
        """
        if self.world_mesh is None:
            raise RuntimeError("World mesh not yet initialized. Call `launch` first.")

        dist.barrier(group=process_group)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp.RedOpType | str = ReduceOp.SUM,
        group: ProcessGroup | None = None,
        async_op: bool = False,
    ):
        """
        All-reduce the given tensor across all processes in the specified process group.

        This method performs an all-reduce operation on the input tensor, summing its values across all processes.
        """
        if self.world_mesh is None:
            raise RuntimeError("World mesh not yet initialized. Call `launch` first.")

        dist.all_reduce(
            tensor, op=cast(dist.ReduceOp.RedOpType, op), group=group, async_op=async_op
        )

    def set_pg_timeouts(self, timeout: dt.timedelta):
        """
        Sets the timeout for all PGs in the provided mesh, and the default (world) group.

        Note: synchronizes via a barrier, before changing the timeouts. This is important, because
        otherwise you may face a race where the slow rank has not reached the timeout reduction point
        yet due to slow operations permitted under the old timeout value, but other faster ranks may
        start issuing collectives under the new shorter timeout and then immediately timeout.
        """
        if self.world_mesh is None:
            raise RuntimeError("World mesh not yet initialized. Call `launch` first.")

        # Ensure that all the ranks have reached the point of setting the new timeout-
        # otherwise, some ranks may issue collectives with the new/shorter timeout and
        # those may time out, before other ranks have finished with initialization done
        # under the old/slow timeout.
        dist.barrier(device_ids=[self.device_module.current_device()])
        self.device_module.synchronize()

        groups = [
            self.world_mesh.get_group(mesh_dim) for mesh_dim in range(self.world_mesh.ndim)
        ]

        # None represents the 'default' PG, not part of the mesh
        groups.append(None)  # type: ignore
        for group in groups:
            c10d._set_pg_timeout(timeout, group)

        logger.info(f"Set process group timeout to {timeout}")

    #######################
    # Gradient Norm Utils #
    #######################

    @torch.no_grad()
    def get_total_norm(
        self,
        parameters: torch.Tensor | Iterable[torch.Tensor],
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
        async_op: bool = False,
    ) -> torch.Tensor:
        """
        Get the total gradient norm of an iterable of parameters.

        Gradient norm clipping requires computing the gradient norm over the entire model.
        `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
        We need to manually reduce the gradient norm across PP stages.

        Args:
            parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).

        """
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        if (pp_mesh := self.get_mesh("pp")) is not None:
            if math.isinf(norm_type):
                dist.all_reduce(
                    total_norm,
                    op=dist.ReduceOp.MAX,
                    group=pp_mesh.get_group(),
                    async_op=async_op,
                )
            else:
                total_norm **= norm_type
                dist.all_reduce(
                    total_norm,
                    op=dist.ReduceOp.SUM,
                    group=pp_mesh.get_group(),
                    async_op=async_op,
                )
                total_norm **= 1.0 / norm_type

        return total_norm
