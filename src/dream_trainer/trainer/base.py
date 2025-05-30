import contextlib
import datetime as dt
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from itertools import repeat
from typing import TYPE_CHECKING, Any, Iterable

import dist_util.ops as dist_ops
import torch
import torch.nn as nn
from torch.distributed._composable.replicate import DDP as DDPModule
from torch.distributed.fsdp import FSDPModule
from torch.optim.optimizer import Optimizer
from typing_extensions import override

from dream_trainer.configs.trainer import TrainingParameters
from dream_trainer.utils import logger
from dream_trainer.utils.common import seed_everything, stacked_context
from dream_trainer.utils.dataloader import (
    get_train_dataloader_steps,
    get_val_dataloader_steps,
)

from .abstract import AbstractTrainer, AbstractTrainerConfig

if TYPE_CHECKING:
    from dream_trainer.callbacks import CallbackCollection


@dataclass(kw_only=True)
class BaseTrainerConfig(AbstractTrainerConfig):
    training_parameters: TrainingParameters
    callbacks: "CallbackCollection | None" = None


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

    # Iterators
    _train_iterator: Iterable[Any]
    _val_iterator: Iterable[Any]

    def __init__(self, config: BaseTrainerConfig, *args, **kwargs) -> None:
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
        pass

    @abstractmethod
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        pass

    #######################
    # Convenience Methods #
    #######################

    def eval(self):
        self.training = False
        for model in self.named_models().values():
            model.eval()
        self.world.barrier()

    def train(self):
        self.training = True
        for model in self.named_models().values():
            if any(p.requires_grad for p in model.parameters()):
                model.train()
        self.world.barrier()

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
    def no_gradient_sync(self, model: nn.Module):
        """
        Disable gradient sync during accumulation steps
        and mark the final backward for FSDP.

        Usage:
            with self.no_gradient_sync(self.model):
                loss.backward()
        """
        if self.world.world_size == 1:
            # If in single process environment, don't sync gradients
            yield
            return

        assert isinstance(model, (FSDPModule, DDPModule)), (
            f"Expected FSDPModule or DDPModule, got {type(model).__name__}"
        )

        is_last_backward = not self.is_accumulating_gradients

        # Synchronize gradients only when done accumulating gradients
        model.set_requires_gradient_sync(is_last_backward)

        if isinstance(model, FSDPModule):
            # FSDP models set is_last_backward when done accumulating gradients
            model.set_is_last_backward(is_last_backward)

        try:
            yield
        finally:
            # Always restore is_last_backward when we set it
            if is_last_backward and isinstance(model, FSDPModule):
                model.set_is_last_backward(False)

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
        return (
            (self.local_batches + 1) % self._num_gradient_accumulation_steps != 0
        ) and not self._is_last_training_batch

    ######################
    # Model Fitting Loop #
    ######################

    def perform_training_epoch(self):
        if self._num_train_steps <= 0:
            return

        self.train()
        self.callbacks.pre_train_epoch(self)

        for batch_idx, batch in zip(range(self._num_train_steps), self._train_iterator):
            self._is_last_training_batch = batch_idx == self._num_train_steps - 1

            # Train Step
            self.callbacks.pre_train_step(self, batch_idx)
            with stacked_context(
                [self.world.train_context()] + self.callbacks.train_context(self)
            ):
                result = self.training_step(batch, batch_idx)

            self.callbacks.post_train_step(self, result, batch_idx)
            self.local_batches += 1

            if not self.is_accumulating_gradients:
                self._local_step += 1
                self.global_step += 1

            # Reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if self._local_step == 1:
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

        self.callbacks.post_train_epoch(self, result)

    @torch.no_grad()
    def perform_validation_epoch(self):
        if self._num_val_steps <= 0 or self._val_iterator is None:
            return

        self.eval()

        # Validation Epoch Start

        self.callbacks.pre_validation_epoch(self)

        # Validation Epoch Loop
        for batch_idx, batch in zip(range(self._num_val_steps), self._val_iterator):
            self.callbacks.pre_validation_step(self, batch_idx)

            with stacked_context(self.callbacks.validation_context(self)):
                result = self.validation_step(batch, batch_idx)

            self.callbacks.post_validation_step(self, result, batch_idx)

        # Validation Epoch End
        self.callbacks.post_validation_epoch(self, result)

    def perform_sanity_validation_steps(self):
        # Don't perform sanity validation on resumption
        if self.current_epoch > 0:
            return

        num_val_steps = self._num_val_steps
        self._num_val_steps = self._num_sanity_val_steps
        self.perform_validation_epoch()
        self._num_val_steps = num_val_steps

    def _setup_trainer_metadata(self):
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

        # Create iterators
        self._train_iterator = iter(self.train_dataloader)
        self._val_iterator = iter(self.val_dataloader)

    def _fit(self):
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
