import queue
import shutil
import threading
import time
import warnings
from concurrent.futures import Future
from typing import Any

import torch.distributed.checkpoint as dcp
from typing_extensions import override

from dream_trainer.configs import CheckpointParameters
from dream_trainer.utils import logger

from .base import CheckpointCallback
from .types import Checkpoint
from .utils import find_checkpoints

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.",
)


class Terminate:
    pass


class AsyncCheckpointCallback(CheckpointCallback):
    _save_future: Future | None
    _load_future: Future | None

    def __init__(self, config: CheckpointParameters):
        super().__init__(config)

        self._save_future = None
        self._load_future = None

        self.purge_queue = queue.Queue()
        self.purge_thread = threading.Thread(
            target=_purge_thread, args=(self.purge_queue,), daemon=True
        )

    @override
    def _save(self, checkpoint: Checkpoint):
        self._wait_save()  # wait for previous save to finish

        logger.info(f"Saving checkpoint {checkpoint.checkpoint_id}")
        self._save_future = dcp.state_dict_saver.async_save(
            self.trainer.state_dict(),
            checkpoint_id=str(self.root_dir / checkpoint.checkpoint_id),
            process_group=self.pg,
        )
        self._save_future.add_done_callback(
            lambda _: logger.info(
                f"Saved checkpoint to {self.root_dir / checkpoint.checkpoint_id}"
            )
        )
        self._save_future.add_done_callback(self._cleanup_checkpoints)

    @override
    def _load(self, checkpoint: Checkpoint, state_dict: dict[str, Any]):
        future = Future[None]()

        def _worker():
            try:
                logger.info(f"Loading checkpoint {checkpoint.checkpoint_id}")
                dcp.state_dict_loader.load(
                    state_dict,
                    checkpoint_id=str(self.root_dir / checkpoint.checkpoint_id),
                    process_group=self.pg,
                )
                logger.info(f"Resumed {self.trainer.experiment} from step {checkpoint.step}")
                future.set_result(None)
            except Exception as e:
                future.set_exception(e)

        threading.Thread(target=_worker, daemon=True).start()
        self._load_future = future

    @override
    def _cleanup_checkpoints(self):
        checkpoints = find_checkpoints(self.root_dir, self.config.resume_mode)
        purge_checkpoints = checkpoints[self.config.keep_top_k :]

        for checkpoint in purge_checkpoints:
            self.purge_queue.put(str(self.root_dir / checkpoint.checkpoint_id))

    ##################
    # Callback Hooks #
    ##################

    def post_setup(self):
        super().post_setup()

        # async purging
        if self.config.keep_top_k > 0:
            self.purge_thread.start()

    def pre_validation_step(self, batch: dict[str, Any], batch_idx: int):
        if batch_idx == 0:
            self._wait_load()

    def pre_train_step(self, batch: dict[str, Any], batch_idx: int):
        if batch_idx == 0:
            self._wait_load()

    def on_interrupt(self):
        self._close()

    ##############
    # Lifecycle #
    ##############

    def _wait_save(self) -> None:
        if self._save_future is not None:
            self._save_future.result()
            self._save_future = None

    def _wait_load(self) -> None:
        if self._load_future is not None:
            self._load_future.result()
            self._load_future = None

    def _close(self):
        logger.info("Closing AsyncCheckpointCallback. Saving any remaining checkpoints.")
        self._wait_save()
        self._wait_load()

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
            assert isinstance(path, str)
            logger.debug("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)  # TODO:  Work with cloud storage
            logger.debug(
                "Checkpoint deleted %s in %.2f seconds.", path, time.monotonic() - begin
            )
    finally:
        logger.debug("Destroying the purge thread.")
