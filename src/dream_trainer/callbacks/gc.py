import gc
from typing import override

from dream_trainer.utils import logger

from ..trainer import BaseTrainer
from .callback import Callback


class GarbageCollectionCallback(Callback[BaseTrainer]):
    """
    Takes control of garbage collection to avoid stragglers.
    """

    def __init__(self, gc_every_n_steps: int = 1000):
        assert gc_every_n_steps > 0, "gc_every_n_steps must be a positive integer"
        self.gc_every_n_steps = gc_every_n_steps

    def collect(self, reason: str):
        gc.collect(generation=1)
        logger.debug(f"[GC] {reason}")

    @override
    def pre_launch(self):
        gc.disable()
        self.collect("Initial GC collection")

    @override
    def pre_train_step(self, _, batch_idx: int):
        if batch_idx % self.gc_every_n_steps == 0:
            self.collect("GC collection invoked by train step")

    @override
    def pre_train_epoch(self):
        self.collect("GC collection invoked by train epoch")

    @override
    def post_train_epoch(self, _):
        self.collect("GC collection invoked by train epoch")

    @override
    def pre_validation_epoch(self):
        self.collect("GC collection invoked by validation epoch")

    @override
    def post_validation_epoch(self, _):
        self.collect("GC collection invoked by validation epoch")
