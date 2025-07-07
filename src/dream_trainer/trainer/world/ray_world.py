import datetime as dt
import os

import torch
import torch._inductor.config
import torch.distributed
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import override

from dream_trainer.configs import DeviceParameters
from dream_trainer.utils import logger
from dream_trainer.utils.common import get_device_info

from .distributed_world import DistributedWorld


class RayDistributedWorld(DistributedWorld):
    def __init__(self, config: DeviceParameters):
        config.validate()
        self.config = config

        self.world_mesh: DeviceMesh | None = None
        self.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

        # NOTE: `device_module.set_device` has to be set before creating TorchFT manager.
        self.device_type, self.device_module = get_device_info()
        self.device = torch.device(f"{self.device_type}:{int(os.environ['LOCAL_RANK'])}")
        self.device_module.set_device(self.device)

    #########
    # Setup #
    #########

    @override
    def _init_collective(self):
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
