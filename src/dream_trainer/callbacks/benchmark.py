"""
Low-overhead benchmarking callback for continuous monitoring of training speed.

Uses CUDA events for accurate GPU timing with minimal overhead.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, override

import torch
import torch.nn as nn
from torch.optim import Optimizer

from dream_trainer.trainer import BaseTrainer
from dream_trainer.utils import logger

from .callback import Callback


@dataclass
class RunningStats:
    """Tracks running statistics with a sliding window."""

    name: str
    window_size: int = 100
    _samples: deque = field(default_factory=deque)
    bubble_count: int = 0

    def __post_init__(self):
        self._samples = deque(maxlen=self.window_size)

    def update(self, value: float):
        self._samples.append(value)
        if value - min(self._samples) >= 100.0:
            self.bubble_count += 1

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def mean(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def std(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self._samples) / (len(self._samples) - 1)
        return variance**0.5

    @property
    def min(self) -> float:
        if not self._samples:
            return 0.0
        return min(self._samples)

    @property
    def max(self) -> float:
        if not self._samples:
            return 0.0
        return max(self._samples)

    @property
    def last(self) -> float:
        if not self._samples:
            return 0.0
        return self._samples[-1]

    def reset(self):
        self._samples.clear()
        self.bubble_count = 0
        self._in_bubble = False

    def format(self, steps: int, unit: str = "ms") -> str:
        if not self._samples:
            return f"{self.name:<12}: N/A"

        std_field = f"{self.std:8.2f}"
        if self.std >= 100:
            std_field = f"<red>{std_field}</red>"
        elif self.std >= 10:
            std_field = f"<yellow>{std_field}</yellow>"

        bubbles_field = f"{self.bubble_count:>4d}/{steps}"

        return (
            f"{self.name:<12}: {self.mean:8.2f} ± {std_field} {unit:<2} "
            f"[min: {self.min:8.2f}, max: {self.max:8.2f}, last: {self.last:8.2f}] "
            f"bubbles: {bubbles_field}"
        )


class BenchmarkCallback(Callback[BaseTrainer]):
    """
    Low-overhead benchmarking callback that tracks forward, backward, and optimizer step times.

    Uses CUDA events for accurate GPU timing with minimal overhead compared to full profiling.

    Args:
        skip: Number of training steps to skip before collecting timing data (warmup).
        print_every: Print statistics every N training steps. Set to 0 to disable periodic printing.
        window_size: Size of the sliding window for running statistics.
    """

    def __init__(
        self,
        skip: int = 5,
        print_every: int = 10,
        window_size: int = 100,
    ):
        self.skip = skip
        self.print_every = print_every
        self.window_size = window_size

        # Running statistics
        self.forward_stats = RunningStats("Forward", window_size)
        self.backward_stats = RunningStats("Backward", window_size)
        self.optimizer_stats = RunningStats("Optimizer", window_size)
        self.total_step_stats = RunningStats("Total Step", window_size)

        # CUDA events for timing
        self._step_start_event: torch.cuda.Event | None = None
        self._forward_end_event: torch.cuda.Event | None = None
        self._backward_end_event: torch.cuda.Event | None = None
        self._optimizer_start_event: torch.cuda.Event | None = None
        self._optimizer_end_event: torch.cuda.Event | None = None

        # State tracking
        self._steps_seen = 0
        self._collecting = False

        # Original methods storage
        self._original_backward: Any = None

    @override
    def pre_launch(self):
        # Disable sanity validation for benchmarking
        self.trainer.training_parameters.num_sanity_val_steps = 0

    @override
    def pre_fit(self):
        # Wrap the backward method to measure forward vs backward timing
        self._original_backward = self.trainer.backward

        def timed_backward(loss: torch.Tensor, **kwargs):
            # Record end of forward pass / start of backward pass
            if self._collecting and self._forward_end_event is not None:
                self._forward_end_event.record()

            result = self._original_backward(loss, **kwargs)

            # Record end of backward pass
            if self._collecting and self._backward_end_event is not None:
                self._backward_end_event.record()

            return result

        self.trainer.backward = timed_backward

    @override
    def post_fit(self):
        # Restore original backward method
        if self._original_backward is not None:
            self.trainer.backward = self._original_backward

        # Print final statistics
        self._print_stats(final=True)

    @override
    def pre_train_step(self, batch: dict[str, Any], batch_idx: int):
        self._steps_seen += 1
        self._collecting = self._steps_seen > self.skip

        if not self._collecting:
            return

        # Create CUDA events for this step
        self._step_start_event = torch.cuda.Event(enable_timing=True)
        self._forward_end_event = torch.cuda.Event(enable_timing=True)
        self._backward_end_event = torch.cuda.Event(enable_timing=True)

        # Record step start
        self._step_start_event.record()

    @override
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        if not self._collecting:
            if self._steps_seen == self.skip:
                logger.info(
                    f"Benchmark: Warmup complete ({self.skip} steps), now collecting data"
                )
            return

        # Only process timing after optimizer step (when not accumulating gradients)
        if self.trainer.is_accumulating_gradients:
            return

        # Wait for all GPU operations to complete
        torch.cuda.synchronize()

        # Calculate timings from CUDA events
        if (
            self._step_start_event is not None
            and self._forward_end_event is not None
            and self._backward_end_event is not None
        ):
            forward_time = self._step_start_event.elapsed_time(self._forward_end_event)
            backward_time = self._forward_end_event.elapsed_time(self._backward_end_event)

            self.forward_stats.update(forward_time)
            self.backward_stats.update(backward_time)

        if self._optimizer_start_event is not None and self._optimizer_end_event is not None:
            optimizer_time = self._optimizer_start_event.elapsed_time(self._optimizer_end_event)
            self.optimizer_stats.update(optimizer_time)

        # Calculate total step time
        if self._step_start_event is not None and self._optimizer_end_event is not None:
            total_time = self._step_start_event.elapsed_time(self._optimizer_end_event)
            self.total_step_stats.update(total_time)

        # Print statistics periodically
        if self.print_every > 0 and self.forward_stats.count % self.print_every == 0:
            self._print_stats()

    @override
    def pre_optimizer_step(self, model: nn.Module, optimizer: Optimizer):
        if not self._collecting:
            return

        self._optimizer_start_event = torch.cuda.Event(enable_timing=True)
        self._optimizer_start_event.record()

    @override
    def post_optimizer_step(self, model: nn.Module, optimizer: Optimizer):
        if not self._collecting:
            return

        self._optimizer_end_event = torch.cuda.Event(enable_timing=True)
        self._optimizer_end_event.record()

    def _print_stats(self, final: bool = False):
        """Print current running statistics."""
        if self.forward_stats.count == 0:
            logger.info("Benchmark: No data collected yet")
            return

        steps = max(0, self._steps_seen - self.skip)
        header = (
            "=== Final Benchmark Results ===" if final else f"=== Benchmark (n={steps}) ==="
        )
        logger.opt(colors=True).info(
            "\n".join(
                [
                    header,
                    self.forward_stats.format(steps=steps),
                    self.backward_stats.format(steps=steps),
                    self.optimizer_stats.format(steps=steps),
                    self.total_step_stats.format(steps=steps),
                ]
            )
        )

        # Calculate and show breakdown percentages
        if self.total_step_stats.mean > 0:
            fwd_pct = (self.forward_stats.mean / self.total_step_stats.mean) * 100
            bwd_pct = (self.backward_stats.mean / self.total_step_stats.mean) * 100
            opt_pct = (self.optimizer_stats.mean / self.total_step_stats.mean) * 100
            other_pct = 100 - fwd_pct - bwd_pct - opt_pct
            logger.info(
                f"Breakdown: Forward {fwd_pct:.1f}% | Backward {bwd_pct:.1f}% | "
                f"Optimizer {opt_pct:.1f}% | Other {other_pct:.1f}%"
            )

        if final:
            logger.info("=" * 32)

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return current statistics as a dictionary."""
        return {
            "forward": {
                "mean": self.forward_stats.mean,
                "std": self.forward_stats.std,
                "min": self.forward_stats.min,
                "max": self.forward_stats.max,
            },
            "backward": {
                "mean": self.backward_stats.mean,
                "std": self.backward_stats.std,
                "min": self.backward_stats.min,
                "max": self.backward_stats.max,
            },
            "optimizer": {
                "mean": self.optimizer_stats.mean,
                "std": self.optimizer_stats.std,
                "min": self.optimizer_stats.min,
                "max": self.optimizer_stats.max,
            },
            "total_step": {
                "mean": self.total_step_stats.mean,
                "std": self.total_step_stats.std,
                "min": self.total_step_stats.min,
                "max": self.total_step_stats.max,
            },
        }
