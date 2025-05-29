from dataclasses import dataclass, field
from typing import Any

import torch

from dream_trainer.configs.logger import LoggingParameters
from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig


@dataclass(kw_only=True)
class LoggerConfigMixin(AbstractTrainerConfig):
    logging_parameters: LoggingParameters = field(default_factory=LoggingParameters)


class LoggerMixin(AbstractTrainer):
    config: LoggerConfigMixin

    def __init__(self, config: LoggerConfigMixin):
        super().__init__(config)

        self.logging_parameters = config.logging_parameters

    def log_config(self, config: dict[str, Any] | Any):
        raise NotImplementedError("Please implement `log_config`")

    def log_dict(self, logs: dict[str, Any], **kwargs: Any):
        """
        Log a dictionary of scalar values (e.g., losses, metrics) to the logger.

        Args:
            logs (dict[str, Any]): Dictionary containing key-value pairs to log.
        """
        raise NotImplementedError("Please implement `log_dict`")

    def log_images(
        self,
        images: torch.Tensor,
        caption: list[str] | None = None,
        desc: str | None = None,
        **kwargs: Any,
    ):
        """
        Log a single image or a batch of images to the logger.

        Args:
            image (torch.Tensor): Image tensor of shape (B, C, H, W).
            caption (list[str] | None): Optional list of captions for each image.
            desc (str | None): Description/name for the logged images.
        """
        raise NotImplementedError("Please implement `log_image`")

    def log_videos(
        self,
        videos: torch.Tensor,
        caption: list[str] | None = None,
        desc: str | None = None,
        **kwargs: Any,
    ):
        """
        Log a single video or a batch of videos to the logger.

        Args:
            video (torch.Tensor): Video tensor of shape (B, C, T, H, W).
            caption (list[str] | None): Optional list of captions for each video.
            desc (str | None): Description/name for the logged videos.
        """
        raise NotImplementedError("Please implement `log_video`")

    def log_plot(self, plot: Any, desc: str | None = None, **kwargs: Any):
        """
        Log a plot to the logger.
        """
        raise NotImplementedError("Please implement `log_plot`")

    def log_table(self, table: Any, desc: str | None = None, **kwargs: Any):
        """
        Log a table to the logger.
        """
        raise NotImplementedError("Please implement `log_table`")

    def log_code(self, directory: str = "./", **kwargs: Any):
        """
        Log all files in a directory to the logger.
        """
        raise NotImplementedError("Please implement `log_code`")
