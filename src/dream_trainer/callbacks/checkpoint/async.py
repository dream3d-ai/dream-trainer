import queue
import shutil
import threading
import time
from concurrent.futures import Future

import torch.distributed.checkpoint as dcp
from typing_extensions import override

from dream_trainer.trainer.abstract import AbstractTrainer
from dream_trainer.utils import logger

from .base import CheckpointCallback
from .types import Checkpoint
from .utils import find_checkpoints


class Terminate:
    pass


class AsyncCheckpointCallback(CheckpointCallback):
    _save_future: Future | None

    @override
    def _save(self, trainer: AbstractTrainer, checkpoint: Checkpoint):
        self._async_wait()  # wait for previous save to finish
        self._save_future = dcp.async_save(  # type: ignore
            trainer.state_dict(),
            checkpoint_id=str(self._checkpoint_dir / checkpoint.checkpoint_id),
            process_group=self.pg,
        )

    @override
    def _cleanup_checkpoints(self):
        if self.config.keep_top_k == 0:
            return

        checkpoints = find_checkpoints(self._checkpoint_dir, self.config.resume_mode)
        purge_checkpoints = checkpoints[-self.config.keep_top_k :]

        for checkpoint in purge_checkpoints:
            self.purge_queue.put(str(self._checkpoint_dir / checkpoint.checkpoint_id))

    # -------------------------------------------------------------------------
    # Callback Hooks
    # -------------------------------------------------------------------------

    def post_setup(self):
        super().post_setup()

        # async purging
        if self.config.keep_top_k > 0:
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=_purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()

    def post_fit(self):
        self._close()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _async_wait(self) -> None:
        if self.async_future is not None:
            self.async_future.result()
            self.async_future = None

    def _close(self):
        logger.info("Closing AsyncCheckpointCallback. Saving any remaining checkpoints.")
        self._async_wait()

        if self.config.keep_top_k > 0:
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
            assert isinstance(path, str)
            logger.debug("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)  # TODO:  Work with cloud storage
            logger.debug(
                "Checkpointer deleted %s in %.2f seconds.", path, time.monotonic() - begin
            )
    finally:
        logger.debug("Destroying the purge thread.")
