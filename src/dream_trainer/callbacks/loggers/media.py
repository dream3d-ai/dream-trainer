from itertools import chain, repeat
from typing import Annotated, Any

import torch
from torch import Tensor
from typing_extensions import override

from dream_trainer.trainer.mixins.loggers import LoggerMixin

from ..callback import RankZeroCallback


def validate_shape(shape: tuple[int, ...]):
    if len(shape) not in (4, 5):
        raise ValueError(f"Expected a batch of images (4D) or videos (5D). Got shape: {shape}")

    if shape[1] not in (1, 3):
        raise ValueError(
            f"Shape must be of the form (B, C, H, W) or (B, C, T, H, W) with C either 1 or 3. Got shape: {shape}"
        )
    return True


def format_batch(batch: Tensor | list[torch.Tensor], spacing: int = 2) -> torch.Tensor:
    """
    Formats a batch of images or videos for processing.

    Args:
        batch (Tensor | list[Tensor]): The batch of samples to format.
        spacing (int): The spacing to include between samples in the batch.

    Returns:
        Tensor: The sample if a single sample is given. Otherwise, the list of samples are
        merged into a single sample by concatenation along their width (trailing) dimension.

    Raises:
        TypeError: If the sample type in the batch is unsupported.
        ValueError: If the samples have inconsistent or unexpected shapes.
    """
    match batch:
        case Tensor():
            validate_shape(batch.shape)
            return batch
        case list() | tuple() if all(isinstance(b, torch.Tensor) for b in batch):
            assert all(validate_shape(b.shape) for b in batch)
            assert all(b.shape == batch[0].shape for b in batch), (
                "All tensors must have same shape"
            )

            # Create black spacers to be interleaved between samples
            shape = batch[0].shape[:-1] + (spacing,)
            space = batch[0].new_full(shape, min(b.min().item() for b in batch))

            # Interleave samples and batches
            batch = torch.cat([sample for sample in chain(*zip(batch, repeat(space)))], dim=-1)
            return batch[..., :-spacing]
        case _:
            raise TypeError(
                f"Unsupported sample type: {type(batch)}. Please provide either a single sample or tuple of samples"
            )


class MediaLoggerCallback(RankZeroCallback[LoggerMixin]):
    """
    A callback that logs image or video samples and captions to the trainer.

    If a single batch is given, log_images or log_videos is called on the batch.
    If a list of batches is given, we concatenate samples along the width (trailing) dimension
    to convert it into a single batch.

    # TODO: Support logging more types of media (audio, ...)
    """

    def __init__(
        self,
        num_samples: int = 32,
        spacing: int = 2,
        description: str = "samples",
        image_key: str = "samples",
        caption_key: str | None = None,
    ):
        self.num_samples = num_samples
        self.spacing = spacing
        self.description = description
        self.image_key = image_key
        self.caption_key = caption_key

    @override
    def pre_validation_epoch(self):
        self._samples: list[tuple[Tensor, str]] = []

    @override
    def post_validation_step(self, result: dict[str, Any], batch_idx: int):
        if len(self._samples) >= self.num_samples:
            return

        if self.image_key not in result:
            raise ValueError(
                f"Please ensure the key `{self.image_key}` is present in the validation step result dictionary to use the MediaLoggerCallback."
            )

        if self.caption_key is not None and self.caption_key not in result:
            raise ValueError(
                f"Please ensure caption_key is None or the key `{self.caption_key}` is present in the validation step result dictionary to use the MediaLoggerCallback."
            )

        samples: Annotated[Tensor, "B C H W | B C T H W"] = result.pop(self.image_key)
        captions: list[str] = (
            [""] * len(samples) if self.caption_key is None else result.pop(self.caption_key)
        )

        batch = list(zip(format_batch(samples, self.spacing), captions))
        self._samples.extend(batch[: self.num_samples - len(self._samples)])

    @override
    def post_validation_epoch(self, *_):
        image_samples = [sample for sample in self._samples if sample[0].ndim == 3]
        video_samples = [sample for sample in self._samples if sample[0].ndim == 4]

        assert len(image_samples) + len(video_samples) == len(self._samples) <= self.num_samples

        if image_samples:
            images, image_captions = zip(*image_samples)
            self.trainer.log_images(images, image_captions)
        if video_samples:
            videos, video_captions = zip(*video_samples)
            self.trainer.log_videos(videos, video_captions)
