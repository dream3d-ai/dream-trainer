from dream_trainer.trainer.mixins.loggers import LoggerMixin

from ..callback import RankZeroCallback


class LRLoggerCallback(RankZeroCallback[LoggerMixin]):
    def __init__(self, log_frequency: int = 1):
        """
        Initialize the LearningRateLogger.

        Args:
            log_frequency: Log every N completed gradient updates. Defaults to 1.
        """
        if log_frequency < 1:
            raise ValueError("log_frequency must be positive")
        self.log_frequency = log_frequency
        self._learning_rates: dict[str, float] = {}

    def post_optimizer_step(self, model, optimizer):
        # Capture the rate used by this optimizer before its scheduler advances.
        for name, candidate in self.trainer.named_optimizers().items():
            if candidate is optimizer:
                self._learning_rates[f"learning_rate/{name}"] = optimizer.param_groups[0]["lr"]
                break

    def post_train_step(self, result, batch_idx):
        if self.trainer.is_accumulating_gradients:
            return
        completed_step = self.trainer.global_step + 1
        if self._learning_rates and completed_step % self.log_frequency == 0:
            self.trainer.log_dict(
                {**self._learning_rates, "trainer/global_step": completed_step}
            )
        self._learning_rates.clear()
