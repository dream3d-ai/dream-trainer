from dream_trainer.trainer.mixins.loggers import LoggerMixin

from ..callback import Callback


class LRLoggerCallback(Callback[LoggerMixin]):
    def __init__(self, log_frequency: int = 8):
        """
        Initialize the LearningRateLogger.

        Args:
            log_frequency (int): The number of gradient accumulation steps to skip before logging. Defaults to 1.
        """
        self.log_frequency = log_frequency

    def post_optimizer_step(self, *_):
        if self.trainer.global_step % self.log_frequency == 0:
            log_dict = {
                f"learning_rate/{name}": optimizer.param_groups[0]["lr"]
                for name, optimizer in self.trainer.named_optimizers().items()
            }
            self.trainer.log_dict(log_dict)
