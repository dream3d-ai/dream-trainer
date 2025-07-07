from abc import abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import DDP as DDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining.schedules import _PipelineSchedule
from typing_extensions import override

from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig
from dream_trainer.utils import logger
from dream_trainer.utils.common import configuration_ctx
from dream_trainer.utils.materialize import materialize_distributed_module


@dataclass(kw_only=True)
class ModelSetupConfigMixin(AbstractTrainerConfig): ...


class ModelSetupMixin(AbstractTrainer):
    """
    A Mixing that handles the correct ordering of model configuration, parallelism, compilation,
    and weight initialization.
    """

    config: ModelSetupConfigMixin

    ###########################
    # AbstractTrainer Methods #
    ###########################

    @override
    def named_models(self) -> dict[str, nn.Module]:
        return {name: getattr(self, name) for name in self._model_names}

    @override
    def get_module(self, fqn: str) -> nn.Module:
        model, *submodules = fqn.split(".")
        return getattr(self, model).get_submodule(".".join(submodules))

    ########################
    # User-Defined Methods #
    ########################

    @abstractmethod
    def configure_models(self):
        pass

    def post_configure_models(self):
        pass

    def mark_forward_methods(self) -> list[str]:
        return []

    @abstractmethod
    def init_weights(self):
        pass

    def context_parallel_buffers(self) -> list[torch.Tensor]:
        raise NotImplementedError(
            "Please implement `context_parallel_buffers` in your trainer. Return buffers like freq_cis"
        )

    def apply_pipeline_parallel(
        self, pp_mesh: DeviceMesh
    ) -> dict[str, tuple[_PipelineSchedule, list[nn.Module], bool, bool]]:
        """
        Apply pipeline parallelism to the trainer model.

        Returns:
            dict: {
                attr_name (str): The attribute name of the model part on the trainer,
                value: (
                    pipeline_schedule (_PipelineSchedule): The pipeline schedule for the model,
                    model_parts (list[nn.Module]): The list of model parts (pipeline stages),
                    has_first_stage (bool): Whether this rank has the first pipeline stage,
                    has_last_stage (bool): Whether this rank has the last pipeline stage
                )
            }
        """
        raise NotImplementedError(
            "Please implement `apply_pipeline_parallel` in your trainer or set device_parameters.pipeline_parallel_degree=1"
        )

    def apply_tensor_parallel(self, tp_mesh: DeviceMesh):
        """
        User-defined method to apply tensor parallelism to the model.
        """
        raise NotImplementedError(
            "Please implement `apply_tensor_parallel` in your trainer or set device_parameters.tensor_parallel_degree=1"
        )

    def apply_compile(self):
        raise NotImplementedError(
            "Please implement compile_model or set device_parameters.compile_model=False"
        )

    def apply_activation_checkpointing(self) -> None:
        raise NotImplementedError(
            "Please implement `apply_activation_checkpointing` in your trainer or set training_parameters.checkpoint_activations=False"
        )

    def apply_fully_shard(self, config: dict[str, Any]) -> None:
        raise NotImplementedError(
            "Please implement `apply_fully_shard` or disable all parallelism but dp_replicate"
        )

    def apply_replicate(self, dp_replicate_mesh: DeviceMesh):
        raise NotImplementedError(
            "Please implement `apply_replicate` or use non-DDP DeviceParameters."
            "Ex:\nfrom torch.distributed._composable.replicate import replicate \nreplicate(self.model, device_mesh=self.world.get_mesh('dp_replicate'))"
        )

    #######################
    # Convenience Methods #
    #######################

    def get_model(self, name: str) -> nn.Module:
        return getattr(self, name)

    def get_submodule(self, name: str) -> nn.Module:
        child_name, *submodule_name = name.split(".", 1)
        return self.get_model(child_name).get_submodule(".".join(submodule_name))

    ###################
    # Private Methods #
    ###################

    def _apply_pipeline_parallel(self):
        if (pp_mesh := self.world.get_mesh("pp")) is not None:
            raise NotImplementedError("Pipeline parallelism not implemented")
            self.apply_pipeline_parallel(pp_mesh)
            logger.info("Applied Pipeline Parallelism")

    def _apply_tensor_parallel(self):
        if (tp_mesh := self.world.get_mesh("tp")) is not None:
            self.apply_tensor_parallel(tp_mesh)
            logger.info("Applied Tensor Parallelism")

    def _apply_activation_checkpointing(self):
        if self.device_parameters.checkpoint_activations:
            self.apply_activation_checkpointing()
            logger.info("Applied Activation Checkpointing")

    def _apply_compile(self):
        if self.device_parameters.compile_model:
            self.apply_compile()
            logger.info("Compiled Model")

    def _apply_fully_shard(self):
        config = self.world.fsdp_config

        if config is not None:
            self.apply_fully_shard(config)
            logger.info("Applied Fully Shard")
        elif (dp_replicate_mesh := self.world.get_mesh("dp_replicate")) is not None:
            self.apply_replicate(dp_replicate_mesh)
            logger.info("Applied Replicate")
        else:
            logger.debug(
                "Skipping Fully Shard & Replicate because dp_shard and dp_replicate are disabled"
            )
            return

        # Ensure all parameters that require grad are wrapped
        wrapped = {
            f"{name}.{module_name}.{param_name}"
            for name, model in self.named_models().items()
            for module_name, module in model.named_modules()
            for param_name, _ in module.named_parameters()
            if isinstance(module, (FSDPModule, DDPModule))
        }

        requires_grad = {
            f"{name}.{param_name}"
            for name, model in self.named_models().items()
            for param_name, param in model.named_parameters()
            if param.requires_grad
        }

        unwrapped = requires_grad - wrapped

        # assert len(unwrapped) == 0, (
        #     "All parameters that require gradients must be wrapped with fully_shard (or replicate if using DDP). "
        #     f"Unwrapped parameters: {unwrapped}"
        # )

    def _collect_parameter_fqns(self, predicate: Callable[[nn.Module], bool]) -> set[str]:
        """
        Collects the FQNs (Fully Qualified Names) of all parameters in modules that match a predicate.

        Args:
            predicate (Callable[[nn.Module], bool]): A function that takes a module and returns True if its parameters should be collected.

        Returns:
            set[str]: A set of parameter FQNs from modules that match the predicate.
        """
        parameter_fqns = set()
        for module_name, module in self.named_models().items():
            if predicate(module):
                for param_name, _ in module.named_parameters():
                    parameter_fqns.add(f"{module_name}.{param_name}")
        return parameter_fqns

    def _materialize_model(self):
        for model in self.named_models().values():
            # NOTE: check if this works with self.checkpoint_parameters.create_seed_checkpoint
            # originally, it seems like self.checkpoint_parameters.create_seed_checkpoint requires cpu as init_device
            cpu_offload = self.device_parameters.cpu_offload and isinstance(model, FSDPModule)
            init_device = "cpu" if cpu_offload else self.world.device_type
            buffer_device = self.world.device_type if cpu_offload else None

            materialize_distributed_module(
                model,
                init_device=init_device,
                buffer_device=buffer_device,
            )

        # TODO: Add warning of not all weights were initialized
        with torch.no_grad():
            self.init_weights()

        for model in self.named_models().values():
            # Set model to eval if no parameters require grad
            if any(p.requires_grad for p in model.parameters()):
                model.train()
            else:
                model.eval()

        logger.info("Materialized Model")

    def _wrap_forward_method(self, method: Callable) -> Callable:
        """
        Wraps a module's forward method with torch.autocast context manager.

        This wrapper enables automatic mixed precision (AMP) during forward passes by using
        torch.autocast with the configured device type and parameter dtype.

        Args:
            method (Callable): The forward method to wrap

        Returns:
            Callable: The wrapped forward method that runs with autocast enabled
        """

        @wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with torch.autocast(
                device_type=self.world.device_type,
                dtype=self.device_parameters.param_dtype,
            ):
                return method(*args, **kwargs)

        return wrapper

    def _mark_forward_methods(self):
        """
        Registers custom forward methods for FSDP modules.

        Args:
            methods (list[str]): List of method names in dot notation, e.g. ["model.generate", "model.submodule.encode"].
                Each string should specify the path to a module and the forward method to register.
                For example, if you have a module named `self.model` with a forward function `generate`
                and a submodule with the forward function `encode`, you would call:
                    self.mark_forward_methods(["model.generate", "model.submodule.encode"])
        """
        from torch.distributed.fsdp import register_fsdp_forward_method

        forward_methods = [f"{name}.forward" for name in self._model_names]
        forward_methods.extend(self.mark_forward_methods())

        for fqn in forward_methods:
            module_path, method_fqn = fqn.rsplit(".", 1)
            submodule = self.get_submodule(module_path)

            if isinstance(submodule, FSDPModule):
                register_fsdp_forward_method(submodule, method_fqn)
            else:
                # wrap the forward methods in the forward_ctx (autocast)
                method = getattr(submodule, method_fqn)
                setattr(submodule, method_fqn, self._wrap_forward_method(method))

    #########################
    # Top-level Model Setup #
    #########################

    def _configure_models(self):
        self._model_names: list[str] = []
        with (
            torch.device("meta"),
            configuration_ctx(self, self._model_names, nn.Module),
        ):
            self.configure_models()

        self.post_configure_models()
        logger.info("Configured Models")

    def _setup_models(self):
        # Apply parallelism
        self._apply_pipeline_parallel()
        self._apply_tensor_parallel()
        self._apply_activation_checkpointing()
        self._apply_compile()
        self._apply_fully_shard()

        # Materialize model & register forward methods
        self._materialize_model()
        self._mark_forward_methods()
        logger.info("Setup Models")
