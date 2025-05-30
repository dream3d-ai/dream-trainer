from typing import cast

import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FSDPModule
from typing_extensions import override

from dream_trainer.configs import FaultToleranceParameters
from dream_trainer.trainer.world import FaultTolerantWorld

from .callback import Callback

try:
    import torchft as ft  # type: ignore # noqa: F401
except ImportError as e:
    raise ImportError(
        "FtTrainer requires the optional dependency 'torchft'. "
        "Please install it with: pip install dream-trainer[torchft]"
    ) from e


class FaultToleranceCallback(Callback):
    def __init__(self, config: FaultToleranceParameters):
        self.config = config

    @override
    def pre_setup(self):
        self.trainer.world = FaultTolerantWorld(
            self.trainer.config.device_parameters, self.config
        )
        self.trainer.world.ft_manager.set_state_dict_fns(
            self.trainer.load_state_dict,
            self.trainer.state_dict,
        )

    @override
    def post_setup(self):
        self.trainer.world = cast(FaultTolerantWorld, self.trainer.world)
        self.trainer.world.ft_manager.set_state_dict_fns(
            self.trainer.load_state_dict,
            self.trainer.state_dict,
        )

        def _hook(output):
            dist.all_reduce(
                output,
                group=cast(FaultTolerantWorld, self.trainer.world).replicate_pg,
                op=ReduceOp.AVG,
            )

        for model in self.trainer.named_models().values():
            if isinstance(model, FSDPModule):
                model.set_all_reduce_hook(_hook)
