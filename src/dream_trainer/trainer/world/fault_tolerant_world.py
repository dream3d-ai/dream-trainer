import datetime as dt

import dist_util
import torch
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import override

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

    def __init__(self, config: DeviceParameters, ft_config: FaultToleranceParameters):
        super().__init__(config)

        self.ft_config = ft_config
        self.group_rank = dist_util.core.get_dist_local_rank()
        self.group_size = dist_util.core.get_dist_local_world_size()

        self.replica_id = f"{self.ft_config.replica_prefix or 'ft'}_{self.group_rank}"

    @property
    @override
    def dp_replicate_enabled(self):
        return True

    @property
    @override
    def dp_size(self):
        return super().dp_size * self.group_size

    @property
    @override
    def dp_rank(self):
        _dp_size = super().dp_size
        _dp_rank = super().dp_rank
        return _dp_size * self.group_rank + _dp_rank

    def __del__(self):
        if hasattr(self, "ft_manager"):
            self.ft_manager.shutdown(wait=False)

    def setup_ft(self):
        self.ft_pg = ft.ProcessGroupNCCL()
        self.ft_transport = PGTransport(
            self.ft_pg,
            timeout=dt.timedelta(seconds=10),
            device=torch.device(self.device_type),
        )

        # TODO: we need to include ft_manager in trainer.state_dict()
        # when using the CheckpointCallback. Make a FtCheckpointCallback class?
        self.ft_manager = ft.Manager(
            pg=self.ft_pg,
            min_replica_size=self.ft_config.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=False,  # TODO: add async quorum
            replica_id=self.replica_id,
            rank=self.group_rank,
            world_size=self.group_size,
            checkpoint_transport=self.ft_transport,
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
