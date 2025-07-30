import os
from typing import override

import torch

from dream_trainer.utils import logger

from ..trainer import BaseTrainer
from .callback import Callback


def trace_handler(prof: torch.profiler.profile, output_dir: str, export_stacks: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Completed profiling. Exporting results to {output_dir}")

    prof.export_chrome_trace(os.path.join(output_dir, "trace.json"))
    logger.info(f"Exported trace to {os.path.join(output_dir, 'trace.json')}")

    if export_stacks:
        prof.export_stacks(
            os.path.join(output_dir, "stacks.txt"), metric="self_cuda_time_total"
        )
        logger.info(f"Exported stacks to {os.path.join(output_dir, 'stacks.txt')}")

    # prof.export_memory_timeline(f"{output_dir}/memory.html", device="cuda")
    output = prof.key_averages(group_by_stack_n=5)
    with open(os.path.join(output_dir, "time_averages.txt"), "w") as f:
        f.write(output.table(sort_by="self_cuda_time_total", row_limit=1000))
    with open(os.path.join(output_dir, "memory_averages.txt"), "w") as f:
        f.write(output.table(sort_by="self_cuda_memory_usage", row_limit=1000))

    print(output.table(sort_by="self_cuda_time_total", row_limit=10))

    exit()


class ProfileCallback(Callback[BaseTrainer]):
    """
    Profiles the trainer's training step
    """

    def __init__(self, profiler: torch.profiler.profile):
        self.profiler = profiler

    @override
    def pre_launch(self):
        # Disable sanity validation for profiling
        self.trainer.training_parameters.num_sanity_val_steps = 0

    @override
    def pre_fit(self):
        original_training_step = self.trainer.training_step

        def profiled_training_step(*args, **kwargs):
            out = original_training_step(*args, **kwargs)
            self.profiler.step()
            return out

        self.trainer.training_step = profiled_training_step
