import contextlib
import datetime as dt
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from itertools import repeat
from typing import TYPE_CHECKING, Any, Iterable, cast

import dist_util.ops as dist_ops
import torch
import torch.nn as nn
from dist_util.ops import apply_to_collection
from torch.distributed._composable.replicate import DDP as DDPModule
from torch.distributed.fsdp import FSDPModule
from torch.optim.optimizer import Optimizer
from typing_extensions import override

from dream_trainer.configs.trainer import TrainingParameters
from dream_trainer.utils import logger
from dream_trainer.utils.common import seed_everything, stacked_context
from dream_trainer.utils.dataloader import (
    Batch,
    get_train_dataloader_steps,
    get_val_dataloader_steps,
)

from .abstract import AbstractTrainer, AbstractTrainerConfig

if TYPE_CHECKING:
    from dream_trainer.callbacks import CallbackCollection


@dataclass(kw_only=True)
class BaseTrainerConfig(AbstractTrainerConfig):
    """
    Configuration class for BaseTrainer.

    This dataclass holds all configuration parameters needed to initialize
    and run a BaseTrainer instance.

    Attributes:
        training_parameters: Configuration for training hyperparameters including
            epochs, batch size, gradient accumulation, validation frequency, etc.
        callbacks: Collection of callbacks to execute during training lifecycle.
            If None, an empty CallbackCollection will be created.
    """

    training_parameters: TrainingParameters
    callbacks: "CallbackCollection" = cast("CallbackCollection", None)


class BaseTrainer(AbstractTrainer):
    """
    An implementation of a basic training loop, taking into account gradient accumulation,
    validation, callbacks, and contains bindings for backwards calls and optimizer steps.
    """

    config: BaseTrainerConfig
    callbacks: "CallbackCollection"

    # Internal State
    _train_batch_size: int
    _num_train_steps: int
    _num_gradient_accumulation_steps: int

    _num_val_steps: int
    _num_sanity_val_steps: int

    def __init__(self, config: BaseTrainerConfig, *args, **kwargs) -> None:
        """
        Initialize the BaseTrainer.

        Args:
            config: Configuration object containing training parameters and callbacks.
            *args: Additional positional arguments passed to parent class.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(config, *args, **kwargs)

        self.training_parameters = config.training_parameters

        if config.callbacks is None:
            from dream_trainer.callbacks import CallbackCollection

            config.callbacks = CallbackCollection()

        self.callbacks = config.callbacks
        self.callbacks.initialize(self)

        self.training = False
        self._local_step = 0

    ###########################
    # AbstractTrainer Methods #
    ###########################

    def state_dict(self) -> dict[str, Any]:
        """
        Return the complete state dictionary of the trainer.

        This method captures the entire training state including:
        - Trainer metadata (global step, current epoch, callbacks state)
        - All model states
        - All optimizer states
        - All scheduler states
        - Dataloader states (if stateful)

        Returns:
            dict[str, Any]: A dictionary containing the complete trainer state
                that can be used to resume training from a checkpoint.
        """
        return {
            "trainer": {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "callbacks": self.callbacks.state_dict(),
            },
            "models": {name: model.state_dict() for name, model in self.named_models().items()},
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.named_optimizers().items()
            },
            "schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in (self.named_schedulers() or {}).items()
            },
            "dataloaders": {
                "train": getattr(self.train_dataloader, "state_dict", lambda: {})(),
                "val": getattr(self.val_dataloader, "state_dict", lambda: {})(),
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """
        Load a complete state dictionary into the trainer.

        This method restores the entire training state from a checkpoint,
        including trainer metadata, model states, optimizer states, scheduler states,
        and dataloader states.

        Args:
            state_dict: Dictionary containing the complete trainer state,
                typically obtained from a previous call to state_dict().
            strict: If True, raises ValueError when state_dict contains keys
                that don't match the current trainer setup. If False, logs
                warnings for mismatched keys instead.

        Raises:
            ValueError: If strict=True and state_dict contains unexpected keys.
        """
        # Load Trainer State
        trainer_state = state_dict.pop("trainer")
        self.global_step = trainer_state.pop("global_step")
        self.current_epoch = trainer_state.pop("current_epoch")
        self.callbacks.load_state_dict(trainer_state.pop("callbacks"), self)

        # Load Model State
        for name, model in self.named_models().items():
            model.load_state_dict(state_dict.pop("models")[name], strict=strict)

        # Load Optimizer State
        for name, optimizer in self.named_optimizers().items():
            optimizer.load_state_dict(state_dict.pop("optimizers")[name])

        # Load Scheduler State
        for name, scheduler in (self.named_schedulers() or {}).items():
            scheduler.load_state_dict(state_dict.pop("schedulers")[name])

        # Load Dataloader State
        dataloader_state = state_dict.pop("dataloaders")
        getattr(self.train_dataloader, "load_state_dict", lambda _: None)(
            dataloader_state["train"]
        )
        getattr(self.val_dataloader, "load_state_dict", lambda _: None)(dataloader_state["val"])

        if state_dict:
            if strict:
                raise ValueError(f"Missing keys in state_dict: {state_dict.keys()}")
            else:
                logger.warning(f"Missing keys in state_dict: {state_dict.keys()}")

    @override
    def fit(self):
        """
        Execute the complete training pipeline.

        This is the main entry point for training. It handles the entire training
        lifecycle including setup, training loops, validation, and cleanup.

        The method ensures proper cleanup by destroying the distributed process
        group in the finally block, even if training is interrupted.
        """
        try:
            self._fit()
        finally:
            # TODO: close the checkpointer

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    ########################
    # User-Defined Methods #
    ########################

    @abstractmethod
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """
        Execute a single training step.

        This method should implement the forward pass, loss computation,
        and backward pass for a single batch of training data.

        Args:
            batch: Dictionary containing the batch data, typically with keys
                like 'input', 'target', etc.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            dict[str, Any]: Dictionary containing at minimum the computed loss
                and any other metrics or values to log.
        """
        pass

    @abstractmethod
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """
        Execute a single validation step.

        This method should implement the forward pass and metric computation
        for a single batch of validation data. No gradients should be computed.

        Args:
            batch: Dictionary containing the batch data, typically with keys
                like 'input', 'target', etc.
            batch_idx: Index of the current batch within the validation epoch.

        Returns:
            dict[str, Any]: Dictionary containing validation metrics and any
                other values to log.
        """
        pass

    #######################
    # Convenience Methods #
    #######################

    def eval(self):
        """
        Set the trainer and all models to evaluation mode.

        This method:
        - Sets the trainer's training flag to False
        - Calls eval() on all registered models

        This should be called before validation or inference to disable
        dropout, batch normalization updates, and other training-specific
        behaviors.
        """
        self.training = False
        for model in self.named_models().values():
            model.eval()

    def train(self):
        """
        Set the trainer and models to training mode.

        This method:
        - Sets the trainer's training flag to True
        - Calls train() on all registered models that have trainable parameters

        Models without any parameters requiring gradients are kept in eval mode
        to avoid unnecessary computation.
        """
        self.training = True
        for model in self.named_models().values():
            if any(p.requires_grad for p in model.parameters()):
                model.train()

    def step(self, model: nn.Module, optimizer: Optimizer) -> torch.Tensor:
        """
        Performs a single optimization step for the given model and optimizer.

        This method:
            - Computes the total gradient norm for all parameters with gradients.
            - Clips gradients to the configured norm.
            - Calls pre- and post-optimizer step callbacks.
            - Performs the optimizer step.
            - Calls pre- and post-optimizer zero_grad callbacks.
            - Zeros the gradients.
            - Steps the learning rate scheduler if one is associated with the optimizer.

        The step is performed with autocast disabled to ensure numerical stability.

        Args:
            model (nn.Module): The model whose parameters are being optimized.
            optimizer (Optimizer): The optimizer used to update the model parameters.

        Returns:
            torch.Tensor: The total norm of the gradients before clipping.
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        total_norm = self.total_gradient_norm(parameters, p=2, foreach=True)
        self.clip_gradient_norm(parameters, total_norm, foreach=True)

        self.callbacks.pre_optimizer_step(self, model, optimizer)
        optimizer.step()
        self.callbacks.post_optimizer_step(self, model, optimizer)

        self.callbacks.pre_optimizer_zero_grad(self, model, optimizer)
        optimizer.zero_grad()
        self.callbacks.post_optimizer_zero_grad(self, model, optimizer)

        if (scheduler := self.get_scheduler_from_optimizer(optimizer)) is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                scheduler.step()

        return total_norm

    @contextlib.contextmanager
    def loss_parallel(self):
        """
        Context manager for loss parallelism.

        This provides a context where loss computation can be parallelized
        with tensor parallelism on dim=-1.

        Yields:
            None: Context manager for loss parallel computation.

        Example:
            with self.loss_parallel():
                loss = self.compute_loss(outputs, targets)
        """
        with self.world.loss_parallel():
            yield

    def backward(self, loss: torch.Tensor):
        """
        Backward pass for loss, with gradient accumulation scaling and autocast disabled.

        This function is intended to be called inside a training step that is already
        wrapped in autocast (mixed precision). We explicitly disable autocast here to
        avoid calling backward in an autocast context, which can cause issues.

        The loss is divided by the number of gradient accumulation steps to ensure
        correct gradient scaling when using gradient accumulation.

        Args:
            loss (torch.Tensor): The computed loss tensor to backpropagate.
        """
        (loss / self._num_gradient_accumulation_steps).backward()

    @contextlib.contextmanager
    def no_gradient_sync(self, *models: nn.Module):
        """
        Disable gradient sync during accumulation steps
        and mark the final backward for FSDP.

        Usage:
            with self.no_gradient_sync(self.model):
                loss.backward()
        """
        if self.world.world_size == 1 or self._num_gradient_accumulation_steps == 1:
            # If no gradient accumulation or in single process environment, don't sync gradients
            yield
            return

        assert all(isinstance(model, (FSDPModule, DDPModule)) for model in models), (
            f"Expected all modules to be FSDPModule or DDPModule, got {[type(model).__name__ for model in models]}"
        )
        distributed_modules = cast(tuple[FSDPModule | DDPModule, ...], models)

        current_accumulation_step = (
            self.local_batches + 1
        ) % self._num_gradient_accumulation_steps

        # Only update flags when transitioning between states
        is_first_accumulation_step = (
            current_accumulation_step == 1 and not self._is_last_training_batch
        )
        is_last_accumulation_step = (
            current_accumulation_step == 0 or self._is_last_training_batch
        )

        if is_first_accumulation_step:
            # Set requires_gradient_sync to False only on first accumulation step (unless last batch)
            for model in distributed_modules:
                model.set_requires_gradient_sync(False)

        # Set is_last_backward to True on second-to-last step OR if it's the last training batch
        if is_last_accumulation_step:
            for model in distributed_modules:
                model.set_requires_gradient_sync(True)
                if isinstance(model, FSDPModule):
                    model.set_is_last_backward(True)

        yield

    @torch.no_grad()
    def total_gradient_norm(
        self,
        parameters: Iterable[torch.Tensor],
        p=2,
        error_if_nonfinite=False,
        foreach: bool | None = None,
    ):
        grads = [param for param in parameters if param.grad is not None]
        return self.world.get_total_norm(
            parameters=grads,
            norm_type=p,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    @torch.no_grad()
    def clip_gradient_norm(
        self,
        parameters: Iterable[torch.Tensor],
        total_norm: torch.Tensor,
        foreach: bool | None = None,
    ):
        if self.training_parameters.gradient_clip_val is None:
            return

        torch.nn.utils.clip_grads_with_norm_(
            parameters=parameters,
            max_norm=self.training_parameters.gradient_clip_val,
            total_norm=total_norm,
            foreach=foreach,
        )

    @property
    def is_accumulating_gradients(self) -> bool:
        """
        Check if currently accumulating gradients.

        Returns True if the current step is a gradient accumulation step
        (i.e., gradients are being accumulated but not yet applied).
        Returns False if this is the step where accumulated gradients
        will be applied, or if we're on the last training batch.

        Returns:
            bool: True if accumulating gradients, False if applying them.
        """
        return (
            (self.local_batches + 1) % self._num_gradient_accumulation_steps != 0
        ) and not self._is_last_training_batch

    ######################
    # Model Fitting Loop #
    ######################

    def train_context(self):
        """
        Create a stacked context manager for training.

        This method combines the world's training context with any additional
        contexts provided by callbacks into a single stacked context manager.

        Returns:
            contextlib.ExitStack: A stacked context manager that applies all
                training-related contexts.
        """
        return stacked_context(
            [self.world.train_context()] + self.callbacks.train_context(self)
        )

    def perform_training_epoch(self):
        """
        Execute a complete training epoch.

        This method:
        1. Sets the trainer to training mode
        2. Iterates through the training dataloader
        3. Executes training steps with gradient accumulation
        4. Manages callbacks before/after each step
        5. Performs validation at specified intervals
        6. Updates process group timeouts after the first step

        The method handles gradient accumulation by only incrementing the
        global step when gradients are applied (not during accumulation).

        Raises:
            RuntimeError: If fewer batches are received than expected, which
                may indicate data loading issues in distributed training.
        """
        if self._num_train_steps <= 0:
            return

        self.train()
        self.callbacks.pre_train_epoch(self)

        batch_idx = 0
        for batch in self.train_dataloader:
            if batch_idx >= self._num_train_steps:
                break

            self._is_last_training_batch = batch_idx == self._num_train_steps - 1

            # Move batch to device, non-blocking
            batch = apply_to_collection(
                cast(Batch, batch),
                function=lambda t: t.to(self.world.device, non_blocking=True),
                dtype=torch.Tensor,
            )

            # Train Step
            self.callbacks.pre_train_step(self, batch, batch_idx)
            with self.train_context():
                result = self.training_step(batch, batch_idx)

            self.callbacks.post_train_step(self, result, batch_idx)
            self.local_batches += 1

            if not self.is_accumulating_gradients:
                self._local_step += 1
                self.global_step += 1
            batch_idx += 1

            # Reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if self._local_step == 1 and not self.is_accumulating_gradients:
                self.world.set_pg_timeouts(
                    timeout=dt.timedelta(
                        seconds=self.device_parameters.comm.train_timeout_seconds,
                    ),
                )

            # Validation Epoch
            if (
                self.global_step
                % int(self._num_train_steps * self.training_parameters.val_frequency)
            ) == 0 and not self.is_accumulating_gradients:
                self.perform_validation_epoch()
                self.train()

        if (batch_idx + 1) < self._num_train_steps:
            raise RuntimeError(
                f"Worker {self.world.world_mesh.get_rank() if self.world.world_mesh is not None else 'unknown'} received fewer training batches than expected. "
                f"Expected {self._num_train_steps} batches, received {batch_idx + 1}"
            )

        self.callbacks.post_train_epoch(self, result)

    @torch.no_grad()
    def perform_validation_epoch(self):
        """
        Execute a complete validation epoch.

        This method:
        1. Sets the trainer to evaluation mode
        2. Disables gradient computation with @torch.no_grad()
        3. Iterates through the validation dataloader
        4. Executes validation steps
        5. Manages callbacks before/after each step and epoch

        All operations are performed without gradient computation for efficiency.

        Raises:
            RuntimeError: If fewer batches are received than expected, which
                may indicate data loading issues in distributed training.
        """
        if self._num_val_steps <= 0:
            return

        self.eval()

        # Validation Epoch Start
        self.callbacks.pre_validation_epoch(self)

        # Validation Epoch Loop
        batch_idx = 0
        for batch in self.val_dataloader:
            if batch_idx >= self._num_val_steps:
                break

            # Move batch to device, non-blocking
            batch = apply_to_collection(
                cast(Batch, batch),
                function=lambda t: t.to(self.world.device, non_blocking=True),
                dtype=torch.Tensor,
            )

            self.callbacks.pre_validation_step(self, batch, batch_idx)

            with stacked_context(self.callbacks.validation_context(self)):
                result = self.validation_step(batch, batch_idx)

            self.callbacks.post_validation_step(self, result, batch_idx)
            batch_idx += 1

        if (batch_idx + 1) < self._num_val_steps:
            raise RuntimeError(
                f"Worker {self.world.world_mesh.get_rank() if self.world.world_mesh is not None else 'unknown'} received fewer validation batches than expected. "
                f"Expected {self._num_val_steps} batches, received {batch_idx + 1}"
            )

        # Validation Epoch End
        self.callbacks.post_validation_epoch(self, result)

    def perform_sanity_validation_steps(self):
        """
        Perform sanity validation steps before training begins.

        This method runs a limited number of validation steps at the start
        of training to ensure the validation pipeline is working correctly.
        It temporarily overrides the number of validation steps with the
        configured number of sanity validation steps.

        Sanity validation is only performed on the first epoch (epoch 0)
        and is skipped when resuming training from a checkpoint.
        """
        # Don't perform sanity validation on resumption
        if self.current_epoch > 0:
            return

        # Store num val steps & temporarily override to num sanity val steps
        num_val_steps = self._num_val_steps
        self._num_val_steps = self._num_sanity_val_steps

        # Call validation epoch normally & restore num val steps
        self.perform_validation_epoch()
        self._num_val_steps = num_val_steps

    def _setup_trainer_metadata(self):
        """
        Initialize trainer metadata from dataloaders and configuration.

        This method:
        1. Calculates training batch size, number of steps, and gradient
           accumulation steps from the training dataloader
        2. Calculates validation steps and sanity validation steps
        3. Verifies global agreement across all distributed ranks for
           these parameters to ensure consistent training

        The metadata is stored as instance variables for use throughout training.

        Raises:
            AssertionError: If any training parameters differ across distributed ranks.
        """
        # Setup dataloader metadata
        (
            self._train_batch_size,
            self._num_train_steps,
            self._num_gradient_accumulation_steps,
        ) = get_train_dataloader_steps(
            self.train_dataloader,
            self.training_parameters.train_steps_per_epoch,
            self.training_parameters.train_batch_size,
            self.world.dp_size,
        )
        self._num_val_steps, self._num_sanity_val_steps = get_val_dataloader_steps(
            self.val_dataloader,
            self.training_parameters.val_steps_per_epoch,
            self.training_parameters.num_sanity_val_steps,
            self.world.dp_size,
        )

        # Check global agreement for training parameters
        assert dist_ops.global_agreement(self._train_batch_size), (
            "`train_batch_size` must be the same across all ranks"
        )
        assert dist_ops.global_agreement(self._num_train_steps), (
            "`num_train_steps` must be the same across all ranks"
        )
        assert dist_ops.global_agreement(self._num_gradient_accumulation_steps), (
            "`num_gradient_accumulation_steps` must be the same across all ranks"
        )

        # Check global agreement for validation parameters
        assert dist_ops.global_agreement(self._num_val_steps), (
            "`num_val_steps` must be the same across all ranks"
        )
        assert dist_ops.global_agreement(self._num_sanity_val_steps), (
            "`num_sanity_val_steps` must be the same across all ranks"
        )

    def _fit(self):
        """
        Internal implementation of the training pipeline.

        This method orchestrates the complete training lifecycle:
        1. Pre-launch callbacks
        2. World launch (distributed setup)
        3. Seed initialization
        4. Configuration and setup
        5. Sanity validation
        6. Main training loop for all epochs
        7. Post-fit callbacks

        Each stage is wrapped with appropriate callbacks and barriers
        to ensure synchronization in distributed training.
        """
        self.callbacks.pre_launch(self)
        self.world.launch()
        seed_everything(self.seed)

        self.callbacks.pre_configure(self)
        self.configure()
        self.callbacks.post_configure(self)
        self.world.barrier()

        self.callbacks.pre_setup(self)
        self.setup()
        self._setup_trainer_metadata()
        self.callbacks.post_setup(self)
        self.world.barrier()

        # Begin Training
        self.callbacks.pre_fit(self)

        # Sanity Validation Steps
        self.perform_sanity_validation_steps()
        self.world.barrier()

        # Fit Loop
        n_epochs = self.training_parameters.n_epochs
        for _ in range(n_epochs) if n_epochs is not None else repeat(0):
            self.callbacks.pre_epoch(self)
            self.perform_training_epoch()  # Validation handled in training epoch
            self.callbacks.post_epoch(self)

            self.current_epoch += 1

        # Fit End
        self.callbacks.post_fit(self)
