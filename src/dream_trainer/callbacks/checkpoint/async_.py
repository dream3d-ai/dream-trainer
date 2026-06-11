import gc
import queue
import shutil
import threading
import time
import warnings
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed import ReduceOp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from typing_extensions import override

from dream_trainer.configs import CheckpointParameters
from dream_trainer.utils import logger

from .base import CheckpointCallback
from .s3 import delete_s3_checkpoint_prefix, is_s3_uri
from .types import Checkpoint

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.",
)


class Terminate:
    pass


def _accelerator_is_available() -> bool:
    accelerator = getattr(torch, "accelerator", None)
    is_available = getattr(accelerator, "is_available", None)
    if not callable(is_available):
        return False

    try:
        return bool(is_available())
    except Exception:
        return False


def _pinned_memory_staging_options() -> StagingOptions:
    return StagingOptions(True, True, True, _accelerator_is_available())


class AsyncCheckpointCallback(CheckpointCallback):
    _save_future: Future | None
    _pending_saved_step: int | None
    _pending_checkpoint_path: str | Path | None
    _stager: DefaultStager | None
    staging_future: Future | None
    staging: bool

    def __init__(self, config: CheckpointParameters):
        if config.async_mode == "disabled":
            raise ValueError(
                "AsyncCheckpointCallback requires async_mode to be 'async' or "
                "'async_with_pinned_mem'. Use CheckpointCallback for disabled "
                "synchronous checkpointing."
            )

        super().__init__(config)

        self._save_future = None
        self._pending_saved_step = None
        self._pending_checkpoint_path = None
        self._stager = None
        self.staging_future = None
        self.staging = False

        self.purge_queue = queue.Queue()
        self.purge_thread = threading.Thread(
            target=_purge_thread, args=(self.purge_queue,), daemon=True
        )

    @override
    def _save(self, checkpoint: Checkpoint, state_dict: dict[str, Any]):
        self._wait_save()  # wait for previous save to finish

        if self._checkpoint_exists(checkpoint):
            return

        save_path = self._checkpoint_path(checkpoint)
        logger.info(f"Saving checkpoint {checkpoint.checkpoint_id}")
        result = self._async_save_state_dict(state_dict, save_path)
        self._save_future = self._upload_future(result)
        self._pending_saved_step = checkpoint.step
        self._pending_checkpoint_path = save_path

    def _async_save_state_dict(
        self, state_dict: dict[str, Any], save_path: str | Path
    ) -> Future | AsyncSaveResponse:
        save_kwargs: dict[str, Any] = {"checkpoint_id": str(save_path)}
        if self._checkpoint_storage is not None:
            checkpoint_id = str(save_path).rstrip("/").rsplit("/", 1)[-1]
            save_kwargs["storage_writer"] = self._checkpoint_storage.writer(checkpoint_id)

        if self.config.async_mode == "async_with_pinned_mem":
            stager = DefaultStager(_pinned_memory_staging_options())
            try:
                result = dcp.state_dict_saver.async_save(
                    state_dict,
                    **save_kwargs,
                    process_group=self.pg,
                    async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                    async_stager=stager,
                )
            except Exception:
                stager.close()
                raise

            if not isinstance(result, AsyncSaveResponse):
                stager.close()
                raise TypeError(
                    "async_with_pinned_mem expected DCP async_save to return "
                    f"AsyncSaveResponse, got {type(result).__name__}"
                )

            self._stager = stager
            self.staging_future = result.staging_completion
            self.staging = True
            return result

        return dcp.state_dict_saver.async_save(
            state_dict,
            **save_kwargs,
            process_group=self.pg,
        )

    def _upload_future(self, result: Future | AsyncSaveResponse) -> Future:
        if isinstance(result, AsyncSaveResponse):
            return result.upload_completion
        return result

    @torch.no_grad()
    @override
    def save(self):
        if self._last_saved_step == self.trainer.global_step:
            return

        if self._pending_saved_step == self.trainer.global_step:
            return

        if self._current_metric is None:
            raise ValueError(f"{self.config.monitor} was not reported in the last epoch")

        self._wait_save()
        self.trainer.world.all_reduce(self._current_metric, op=ReduceOp.AVG)
        checkpoint = Checkpoint(
            step=self.trainer.global_step, metric=self._current_metric.item()
        )

        gc.collect(generation=1)
        self._save(checkpoint, self.trainer.state_dict())
        gc.collect(generation=1)

    @override
    def _cleanup_checkpoints(self):
        if not self.trainer.world.is_global_zero:
            return

        for checkpoint in self._stale_checkpoints():
            checkpoint_path = self._checkpoint_path(checkpoint)
            if is_s3_uri(checkpoint_path):
                self.purge_queue.put((str(checkpoint_path), self.config.storage_options))
            else:
                self.purge_queue.put(str(checkpoint_path))

    ##################
    # Callback Hooks #
    ##################

    @override
    def post_setup(self):
        super().post_setup()

        # async purging
        if self.config.keep_top_k > 0:
            self.purge_thread.start()

    @override
    def post_fit(self):
        super().post_fit()
        self._close()

    @override
    def on_interrupt(self):
        self._close()

    @override
    def pre_train_step(self, batch: dict[str, Any], batch_idx: int):
        self.maybe_wait_for_staging()

    @override
    def pre_optimizer_step(self, model, optimizer):
        self.maybe_wait_for_staging()

    ##############
    # Lifecycle #
    ##############

    def maybe_wait_for_staging(self) -> None:
        if self.config.async_mode == "async_with_pinned_mem" and self.staging:
            assert self.staging_future is not None
            try:
                self.staging_future.result()
            finally:
                self.staging_future = None
                self.staging = False

    def _close_stager(self) -> None:
        if self._stager is not None:
            self._stager.close()
            self._stager = None

    def _wait_save(self) -> None:
        try:
            self.maybe_wait_for_staging()
            if self._save_future is None:
                return

            self._save_future.result()
            logger.info(f"Saved checkpoint to {self._pending_checkpoint_path}")
            self._save_future = None
            self._last_saved_step = self._pending_saved_step
            self._pending_saved_step = None
            self._pending_checkpoint_path = None
            self._cleanup_checkpoints()
        finally:
            if self.config.async_mode == "async_with_pinned_mem":
                self._close_stager()

    def _close(self):
        logger.info(
            "Closing AsyncCheckpointCallback. Waiting for any remaining checkpoint save."
        )
        try:
            self._wait_save()
        finally:
            self._close_purge_thread()
            self._close_stager()

    def _close_purge_thread(self):
        if self.config.keep_top_k > 0 and self.purge_thread.is_alive():
            self.purge_queue.put(Terminate())
            self.purge_thread.join()


def _purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            if isinstance(path, tuple):
                checkpoint_path, storage_options = path
                assert isinstance(checkpoint_path, str)
                logger.debug("Checkpointer is deleting %s.", checkpoint_path)
                begin = time.monotonic()
                delete_s3_checkpoint_prefix(checkpoint_path, storage_options)
                logger.debug(
                    "Checkpoint deleted %s in %.2f seconds.",
                    checkpoint_path,
                    time.monotonic() - begin,
                )
                continue
            assert isinstance(path, str)
            logger.debug("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.debug(
                "Checkpoint deleted %s in %.2f seconds.", path, time.monotonic() - begin
            )
    finally:
        logger.debug("Destroying the purge thread.")
