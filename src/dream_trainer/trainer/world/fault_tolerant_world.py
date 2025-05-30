import datetime as dt
import os

import dist_util
import torch
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from dream_trainer.configs import DeviceParameters, FaultToleranceParameters

from .distributed_world import DistributedWorld, construct_mesh

try:
    import torchft as ft  # type: ignore # noqa: F401
    from torchft.checkpointing.pg_transport import PGTransport  # type: ignore # noqa: F401
    from torchft.process_group import (  # type: ignore # noqa: F401
        ManagedProcessGroup,
        ft_init_device_mesh,
    )
except ImportError:
    raise ImportError(
        "torchft is not installed. Please install it with `pip install dream-trainer[torchft]` to use fault tolerant training."
    )


class FaultTolerantWorld(DistributedWorld):
    replicate_pg: ProcessGroup
    ft_manager: ft.Manager

    def __init__(self, config: DeviceParameters, ft_config: FaultToleranceParameters):
        super().__init__(config)

        if not config.dp_replicate > 1:
            raise ValueError(
                "Fault tolerant training requires dp_replicate > 1 (HSDP). "
                "Please set dp_replicate > 1 in the config."
            )

        self.ft_config = ft_config
        self.replica_id = f"{self.ft_config.replica_prefix or os.environ['TORCHELASTIC_RUN_ID']}_{self.group_rank}"

    def __del__(self):
        if hasattr(self, "ft_manager"):
            self.ft_manager.shutdown(wait=False)

    def setup_ft(self):
        # Get the group world size and rank from environment variables
        group_world_size = dist_util.core.get_dist_local_world_size()
        group_rank = dist_util.core.get_dist_local_rank()

        self.ft_pg = ft.ProcessGroupNCCL()
        self.ft_transport = PGTransport(
            self.ft_pg,
            timeout=dt.timedelta(seconds=10),
            device=torch.device(self.device_type),
        )

        self.ft_manager = ft.Manager(
            pg=self.ft_pg,
            rank=group_rank,
            world_size=group_world_size,
            replica_id=self.replica_id,
            checkpoint_transport=self.ft_transport,
            min_replica_size=self.ft_config.min_replica_size,
            # State dict functions are set by CheckpointCallback
            load_state_dict=None,
            state_dict=None,
        )

        self.replicate_pg = ManagedProcessGroup(self.ft_manager)
        self.replicate_pg.register("dp_replicate")

    def build_mesh(self, device_type: str) -> DeviceMesh:
        self.setup_ft()

        def _init_device_mesh(
            device_type: str, mesh_shape: list[int], mesh_dim_names: list[str]
        ) -> DeviceMesh:
            return ft_init_device_mesh(
                device_type=device_type,
                mesh_shape=mesh_shape,
                mesh_dim_names=mesh_dim_names,
                replicate_dim=mesh_dim_names.index("dp_replicate"),
                manager=self.ft_manager,
            )

        return construct_mesh(self.config, device_type, _init_device_mesh)
