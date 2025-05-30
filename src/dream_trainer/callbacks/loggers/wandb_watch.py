from typing import Literal

from typing_extensions import override

from dream_trainer.trainer.mixins.loggers import WandBLoggerMixin

from ..callback import RankZeroCallback


class ModelWatchCallback(RankZeroCallback[WandBLoggerMixin]):
    """
    A base callback that logs all returned metrics from training and validation steps and
    epochs.

    This callback only logs scalar metrics.
    """

    def __init__(
        self,
        log: Literal["gradients", "parameters", "all"] | None = None,
        log_freq: int = 1000,
    ):
        self.log: Literal["gradients", "parameters", "all"] | None = log
        self.log_freq = log_freq

    @override
    def post_setup(self):
        models = [
            model
            for model in self.trainer.named_models().values()
            if any(p.requires_grad for p in model.parameters())
        ]
        self.trainer.log_model(models, log=self.log, log_freq=self.log_freq)
