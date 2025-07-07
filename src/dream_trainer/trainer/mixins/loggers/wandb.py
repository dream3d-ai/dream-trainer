import fnmatch
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from dream_trainer.configs.logger import WandbLoggingParameters
from dream_trainer.utils.common import background
from dream_trainer.utils.logging import config_to_dict

from .types import LoggerConfigMixin, LoggerMixin

try:
    import wandb  # type: ignore
    import wandb.sdk.wandb_run  # type: ignore

except ImportError as e:
    raise ImportError(
        "wandb is not installed. Please install it with `pip install dream-trainer[wandb]` to use the WandB logger."
    ) from e


def get_gitignore_patterns(path: str) -> list[str]:
    """
    Get the .gitignore patterns for the working directory.

    Returns:
        list[str]: The list of patterns to ignore.
    """
    ignore_patterns = []
    if os.path.exists(path):
        with open(path, "r") as gitignore_file:
            ignore_patterns = [
                line.strip()
                for line in gitignore_file
                if line.strip() and not line.startswith("#")
            ]

    return ignore_patterns


@dataclass(kw_only=True)
class WandBLoggerConfigMixin(LoggerConfigMixin):
    logging_parameters: WandbLoggingParameters = field(default_factory=WandbLoggingParameters)


class WandBLoggerMixin(LoggerMixin):
    config: WandBLoggerConfigMixin

    def __init__(self, config: WandBLoggerConfigMixin):
        super().__init__(config)

        self.logging_parameters = config.logging_parameters
        self._wandb = wandb.init(
            project=self.project,
            group=self.group,
            name=self.experiment,
            id=self.experiment,  # id == name for resumption
            resume="allow",
            mode="online"
            if (self.logging_parameters.enabled and self.world.is_global_zero)
            else "disabled",
        )

        assert self._wandb is not None
        self._wandb.define_metric("trainer/global_step")
        self._wandb.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        # Monkeypatch annoying wandb.Video "Encoding video... " log
        try:
            from wandb.sdk.data_types.video import printer_asyncio

            printer_asyncio.run_async_with_spinner = lambda spinner_printer, text, func: func()
        except ImportError:
            pass

    def log_model(
        self,
        models: list[nn.Module] | None = None,
        log: Literal["gradients", "parameters", "all"] | None = None,
        log_freq: int = 1000,
    ):
        self._wandb.watch(
            models or list(self.named_models().values()),
            log=log,
            log_freq=log_freq,
        )

    @override
    def log_dict(self, logs: dict[str, Any]):
        logs = {
            k: v.float() if isinstance(v, Tensor) and v.dtype == torch.bfloat16 else v
            for k, v in logs.items()
        }
        self._wandb.log({"trainer/global_step": self.global_step, **logs})

    @override
    @background
    def log_images(
        self,
        images: Tensor | list[Tensor],
        captions: list[str] | None = None,
        desc: str = "media",
    ):
        """
        Log images to WandB.

        Args:
            images (Tensor | list[Tensor]): Image tensor of shape (B, C, H, W) ∈ [-1, 1] or list[Tensor] of shape (C, H, W) ∈ [-1, 1]
            captions (list[str] | None): Optional list of captions for each image
        """
        if isinstance(images, Tensor):
            images = list(images.unbind())

        assert all(image.ndim == 3 for image in images), (
            "Images must be 4D tensor (B, C, H, W) or list of 3D tensors (C, H, W)"
        )
        assert all(image.shape[0] in [1, 3] for image in images), (
            "Images must have 1 or 3 channels"
        )

        # Convert to uint8
        images = [(image * 128 + 128).clip(0, 255).byte() for image in images]
        images = [image.detach().cpu() for image in images]

        _images = [
            wandb.Image(image, caption=caption)
            for image, caption in zip(images, captions or [None] * len(images))
        ]
        self._wandb.log({"trainer/global_step": self.global_step, desc: _images})

    @override
    @background
    def log_videos(
        self,
        videos: Tensor | list[Tensor],
        captions: list[str] | None = None,
        desc: str = "media",
    ):
        """
        Log videos to WandB.

        Args:
            videos (Tensor): Video tensor of shape (B, C, T, H, W) ∈ [-1, 1]
            captions (list[str] | None): Optional list of captions for each video
            desc (str): Description/name for the logged videos
        """
        if isinstance(videos, Tensor):
            videos = list(videos.unbind())

        assert all(video.ndim == 4 for video in videos), (
            "Videos must be 4D tensor (B, C, T, H, W) or list of 4D tensors (C, T, H, W)"
        )
        assert all(video.shape[0] in [1, 3] for video in videos), (
            "Videos must have 1 or 3 channels"
        )

        # Convert to uint8 and (C, T, H, W) -> (T, C, H, W)
        videos = [(video * 128 + 128).clip(0, 255).byte() for video in videos]
        videos = [video.permute(1, 0, 2, 3) for video in videos]
        videos = [video.to("cpu", non_blocking=True) for video in videos]

        _videos = [
            wandb.Video(video.numpy(), caption=caption, format="gif")
            for video, caption in zip(videos, captions or [None] * len(videos))
        ]

        self._wandb.log({"trainer/global_step": self.global_step, desc: _videos})

    def _should_ignore(self, ignore_patterns: list[str]):
        def _ignore(path: str, root: str | None = None) -> bool:
            rel_path = os.path.relpath(path, root)
            return any(
                rel_path.startswith(pattern) or fnmatch.fnmatch(rel_path, pattern)
                for pattern in ignore_patterns
            )

        return _ignore

    @override
    @background
    def log_code(
        self,
        directory: str = "./",
        ignore_patterns: list[str] = [],
        gitignore_path: str | None = "./.gitignore",
    ):
        if gitignore_path is not None:
            ignore_patterns += get_gitignore_patterns(gitignore_path)

        self._wandb.log_code(
            root=directory,
            include_fn=lambda path: path.endswith(".py"),
            exclude_fn=self._should_ignore(ignore_patterns),
        )

    def log_config(self, config: dict[str, Any] | Any):
        if not isinstance(config, dict):
            config = config_to_dict(config)

        self._wandb.config.update(config, allow_val_change=True)
