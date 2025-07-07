import time
from typing import Any, MutableMapping

from tqdm import tqdm
from typing_extensions import override

from dream_trainer.trainer import BaseTrainer

from .callback import Callback


def format_seconds(time):
    prefix = ["", "m", "µ", "n"]
    p = ""
    for p in prefix:
        if time > 1:
            break
        time *= 1000

    return f"{time:6.2f}{p}s"


class factorized_tqdm(tqdm):
    """
    For expensive iterables, this keeps track of both the time taken to serve a sample and the
    time taken to complete a loop separately.
    """

    def __init__(self, *args, **kwargs):
        self._serve_time_ema = 0.0
        self._loop_time_ema = 0.0

        self._start_time: float
        self._end_time: float

        self._decimal_output = False

        super().__init__(
            *args,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix} (Loop:{loop_time}, Latency: {serve_time})]",
            **kwargs,
        )

    def __iter__(self):
        iterator = super().__iter__()
        while True:
            serve_start_time = time.perf_counter()
            obj = next(iterator, None)
            serve_end_time = time.perf_counter()

            if obj is None:
                return

            loop_start_time = time.perf_counter()
            yield obj
            loop_end_time = time.perf_counter()

            self._serve_time_ema = (1 - self.smoothing) * (
                serve_end_time - serve_start_time
            ) + self.smoothing * self._serve_time_ema

            self._loop_time_ema = (1 - self.smoothing) * (
                loop_end_time - loop_start_time
            ) + self.smoothing * self._loop_time_ema

    @property
    def format_dict(self) -> MutableMapping[str, Any]:
        d = super().format_dict

        if isinstance(d["n"], float) and not d["n"].is_integer():
            # If a single float is seen, all future items are formatted as floats
            self._decimal_output = True

        d.update(
            serve_time=format_seconds(self._serve_time_ema),
            loop_time=format_seconds(self._loop_time_ema),
            n_fmt=f"{d['n']:.2f}" if self._decimal_output else f"{int(d['n'])}",
        )
        return d


class ProgressBar(Callback[BaseTrainer]):
    """
    A callback that displays a progress bar for the training and validation loops.

    Args:
        smoothing: The smoothing factor for the progress bar.
        metric: The metric to display in the progress bar. If None, no metric is displayed.
    """

    def __init__(self, smoothing: float = 0.1, metric: str | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.metric = metric

    @override
    def pre_fit(self):
        self.epoch_tqdm = tqdm(
            total=self.trainer.training_parameters.n_epochs,
            desc="Epoch",
            position=0,
            leave=True,
            smoothing=self.smoothing,
        )
        self.epoch_tqdm.update(self.trainer.current_epoch)
        self.epoch_tqdm.refresh()

        self.training_tqdm = factorized_tqdm(
            total=self.trainer._num_train_steps
            // self.trainer._num_gradient_accumulation_steps,
            desc="Training",
            position=1,
            leave=True,
            smoothing=self.smoothing,
        )
        self.training_tqdm.update(self.trainer.global_step % self.trainer._num_train_steps)
        self.training_tqdm.refresh()

        self.validation_tqdm = factorized_tqdm(
            total=self.trainer._num_val_steps * self.trainer.world.dp_size,
            desc="Validation",
            position=2,
            leave=True,
            smoothing=self.smoothing,
        )

    @override
    def post_epoch(self):
        self.epoch_tqdm.update(1)

    @override
    def pre_train_epoch(self):
        self.training_tqdm.reset(
            self.trainer._num_train_steps // self.trainer._num_gradient_accumulation_steps
        )
        self.training_tqdm._start_time = time.perf_counter()

    @override
    def pre_train_step(self, *_):
        self.training_tqdm._end_time = time.perf_counter()
        self.training_tqdm._serve_time_ema = self._update_ema(
            self.training_tqdm._serve_time_ema,
            self.training_tqdm._end_time - self.training_tqdm._start_time,
        )

    @override
    def post_train_step(self, result: dict[str, Any], batch_idx: int):
        if self.metric is not None:
            if self.metric in result:
                self.training_tqdm.set_postfix({self.metric: f"{result[self.metric]:.3f}"})
            else:
                self.training_tqdm.set_postfix({})

        self.training_tqdm.update(1 / self.trainer._num_gradient_accumulation_steps)

        self.training_tqdm._start_time = time.perf_counter()
        self.training_tqdm._loop_time_ema = self._update_ema(
            self.training_tqdm._loop_time_ema,
            self.training_tqdm._start_time - self.training_tqdm._end_time,
        )

    @override
    def post_train_epoch(self, *_):
        self.training_tqdm.refresh()

    @override
    def pre_validation_epoch(self):
        self.validation_tqdm.reset(self.trainer._num_val_steps * self.trainer.world.dp_size)
        self.validation_tqdm._start_time = time.perf_counter()

    @override
    def pre_validation_step(self, *_):
        self.validation_tqdm._end_time = time.perf_counter()
        self.validation_tqdm._serve_time_ema = self._update_ema(
            self.validation_tqdm._serve_time_ema,
            self.validation_tqdm._end_time - self.validation_tqdm._start_time,
        )

    @override
    def post_validation_step(self, *_):
        self.validation_tqdm.update(self.trainer.world.dp_size)

        self.validation_tqdm._start_time = time.perf_counter()
        self.validation_tqdm._loop_time_ema = self._update_ema(
            self.validation_tqdm._loop_time_ema,
            self.validation_tqdm._start_time - self.validation_tqdm._end_time,
        )

    @override
    def post_validation_epoch(self, *_):
        self.validation_tqdm.refresh()

    def _update_ema(self, current: float, update: float) -> float:
        return (1 - self.smoothing) * update + self.smoothing * current
