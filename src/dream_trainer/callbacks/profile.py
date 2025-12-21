import os
from typing import override

import torch

from dream_trainer.utils import logger

from ..trainer import BaseTrainer
from .callback import Callback

# Defaults for memory profiling artifacts
DEFAULT_MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000
DEFAULT_MEMORY_TIMELINE_DEVICE = "cuda:0"
DEFAULT_MEMORY_SNAPSHOT_FILENAME = "memory_snapshot.pickle"


def trace_handler(
    prof: torch.profiler.profile,
    output_dir: str,
    export_stacks: bool = True,
    export_memory_timeline: bool = True,
    memory_timeline_device: str = DEFAULT_MEMORY_TIMELINE_DEVICE,
    dump_memory_snapshot: bool = False,
    memory_snapshot_path: str | None = None,
    stop_memory_recording: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Completed profiling. Exporting results to {output_dir}")

    prof.export_chrome_trace(os.path.join(output_dir, "trace.json"))
    logger.info(f"Exported trace to {os.path.join(output_dir, 'trace.json')}")

    if export_stacks:
        prof.export_stacks(
            os.path.join(output_dir, "stacks.txt"), metric="self_cuda_time_total"
        )
        logger.info(f"Exported stacks to {os.path.join(output_dir, 'stacks.txt')}")

    if export_memory_timeline:
        try:
            prof.export_memory_timeline(
                os.path.join(output_dir, "memory_timeline.html"),
                device=memory_timeline_device,
            )
            logger.info(
                f"Exported memory timeline to {os.path.join(output_dir, 'memory_timeline.html')}"
            )
        except Exception as e:
            logger.warning(f"Failed to export memory timeline: {e}")

    output = prof.key_averages(group_by_stack_n=5)
    with open(os.path.join(output_dir, "time_averages.txt"), "w") as f:
        f.write(output.table(sort_by="self_cuda_time_total", row_limit=1000))
    with open(os.path.join(output_dir, "memory_averages.txt"), "w") as f:
        f.write(output.table(sort_by="self_cuda_memory_usage", row_limit=1000))

    print(output.table(sort_by="self_cuda_time_total", row_limit=10))

    # If requested, dump a GPU memory snapshot before the process exits
    if dump_memory_snapshot and torch.cuda.is_available():
        try:
            snapshot_path = memory_snapshot_path or os.path.join(
                output_dir, DEFAULT_MEMORY_SNAPSHOT_FILENAME
            )
            torch.cuda.memory._dump_snapshot(snapshot_path)
            logger.info(f"Exported CUDA memory snapshot to {snapshot_path}")
        except Exception as e:
            logger.warning(f"Failed to dump CUDA memory snapshot: {e}")
        finally:
            if stop_memory_recording:
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception:
                    pass

    exit()


class ProfileCallback(Callback[BaseTrainer]):
    """
    Profiles the trainer's training step and optionally records CUDA memory history
    for offline visualization.
    """

    def __init__(
        self,
        profiler: torch.profiler.profile,
        *,
        memory_snapshot_enabled: bool = False,
        memory_snapshot_max_entries: int = DEFAULT_MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
        memory_snapshot_output_dir: str | None = None,
    ):
        self.profiler = profiler
        self._memory_snapshot_enabled = memory_snapshot_enabled
        self._memory_snapshot_max_entries = memory_snapshot_max_entries
        self._memory_snapshot_output_dir = memory_snapshot_output_dir

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

        # Begin recording CUDA memory history if requested
        if self._memory_snapshot_enabled and torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history(
                    max_entries=self._memory_snapshot_max_entries
                )
                logger.info(
                    f"Started recording CUDA memory history (max_entries={self._memory_snapshot_max_entries})"
                )
            except Exception as e:
                logger.warning(f"Failed to start CUDA memory history recording: {e}")

    @override
    def post_fit(self):
        # Dump a memory snapshot at the end if recording was enabled and we didn't already exit
        if self._memory_snapshot_enabled and torch.cuda.is_available():
            try:
                output_dir = self._memory_snapshot_output_dir or "."
                os.makedirs(output_dir, exist_ok=True)
                snapshot_path = os.path.join(output_dir, DEFAULT_MEMORY_SNAPSHOT_FILENAME)
                torch.cuda.memory._dump_snapshot(snapshot_path)
                logger.info(f"Exported CUDA memory snapshot to {snapshot_path}")
            except Exception as e:
                logger.warning(f"Failed to dump CUDA memory snapshot: {e}")
            finally:
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception:
                    pass
