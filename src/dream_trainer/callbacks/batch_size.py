from .callback import Callback


class BatchSizeSchedulerCallback(Callback):
    """
    This callback schedules the batch size based by adjusting gradient accumulation steps.
    """
