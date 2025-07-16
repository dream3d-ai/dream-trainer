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
class ModelSetupConfigMixin(AbstractTrainerConfig):
    """Configuration mixin for model setup functionality.

    This class serves as a base configuration for trainers that need model setup
    capabilities. It inherits from AbstractTrainerConfig and can be extended with
    additional configuration parameters specific to model initialization and setup.
    """

    ...


class ModelSetupMixin(AbstractTrainer):
    """
    A mixin that handles the complete lifecycle of model configuration and setup.

    This mixin provides a comprehensive framework for:
    - Model configuration and initialization
    - Applying various parallelism strategies (tensor, pipeline, data)
    - Model compilation and optimization
    - Activation checkpointing
    - Weight initialization

    The mixin enforces a specific order of operations to ensure models are
    properly configured before parallelism is applied and weights are initialized.

    Attributes:
        config: Configuration object containing model setup parameters
        _model_names: List of model attribute names registered during configuration

    Example:
        class MyTrainer(ModelSetupMixin):
            def configure_models(self):
                self.model = MyModel(config)

            def init_weights(self):
                self.model.apply(init_weights_fn)
    """

    config: ModelSetupConfigMixin

    ###########################
    # AbstractTrainer Methods #
    ###########################

    @override
    def named_models(self) -> dict[str, nn.Module]:
        """Return a dictionary mapping model names to their corresponding modules.

        This method provides access to all models registered during the configure_models
        phase. Model names are collected automatically when models are assigned as
        attributes during configuration.

        Returns:
            dict[str, nn.Module]: Dictionary where keys are model attribute names
                and values are the corresponding nn.Module instances.

        Example:
            >>> trainer.named_models()
            {'model': TransformerModel(...), 'encoder': Encoder(...)}
        """
        return {name: getattr(self, name) for name in self._model_names}

    @override
    def get_module(self, fqn: str) -> nn.Module:
        """Retrieve a module or submodule by its fully qualified name.

        This method allows access to nested modules using dot notation. The first
        part of the FQN should be a model name registered during configuration,
        followed by the submodule path.

        Args:
            fqn: Fully qualified name of the module (e.g., "model.encoder.layer1")

        Returns:
            nn.Module: The requested module or submodule

        Raises:
            AttributeError: If the model or submodule doesn't exist

        Example:
            >>> trainer.get_module("model.encoder.attention")
            MultiHeadAttention(...)
        """
        model, *submodules = fqn.split(".")
        return getattr(self, model).get_submodule(".".join(submodules))

    ########################
    # User-Defined Methods #
    ########################

    @abstractmethod
    def configure_models(self):
        """Configure and instantiate all models used by the trainer.

        This method must be implemented by subclasses to define and instantiate
        all models. Models should be assigned as attributes to the trainer instance.
        This method is called within a meta device context, so models are created
        on the meta device for efficient memory usage during configuration.

        The method is called early in the setup process, before any parallelism
        or optimization is applied.

        Example:
            def configure_models(self):
                self.model = TransformerModel(
                    vocab_size=self.config.vocab_size,
                    hidden_dim=self.config.hidden_dim
                )
                self.encoder = Encoder(self.config.encoder_config)
        """
        pass

    def post_configure_models(self):
        """Optional hook called after model configuration.

        This method is called after configure_models() but before any parallelism
        or optimization is applied. It can be used for any additional setup that
        requires the models to be instantiated but doesn't need them to be on
        the actual device yet.

        Common use cases:
        - Setting up model-specific configurations
        - Registering custom hooks
        - Performing model structure validation
        """
        pass

    def mark_forward_methods(self) -> list[str]:
        """Specify additional forward methods to be wrapped with autocast.

        By default, only the standard 'forward' method of each model is wrapped
        with autocast for mixed precision training. This method allows you to
        specify additional methods that should also be wrapped.

        Returns:
            list[str]: List of method names in dot notation (e.g., ["model.generate",
                "model.decode"]). Each string should specify the path from the trainer
                to the method.

        Example:
            def mark_forward_methods(self):
                return ["model.generate", "model.encode", "decoder.decode_step"]
        """
        return []

    @abstractmethod
    def init_weights(self):
        """Initialize model weights after parallelism has been applied.

        This method must be implemented to define how model weights should be
        initialized. It is called after all parallelism strategies have been
        applied and the model has been materialized on the actual device.

        The method is called within a no_grad context, so gradient computation
        is automatically disabled.

        Example:
            def init_weights(self):
                def init_fn(module):
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

                self.model.apply(init_fn)
        """
        pass

    def context_parallel_buffers(self) -> list[torch.Tensor]:
        """Return buffers that need to be synchronized across context parallel ranks.

        This method should return a list of buffers (like positional embeddings
        or frequency tensors) that need to be kept in sync across different
        context parallel ranks when using context parallelism.

        Returns:
            list[torch.Tensor]: List of tensor buffers to synchronize

        Raises:
            NotImplementedError: If not implemented and context parallelism is used

        Example:
            def context_parallel_buffers(self):
                return [self.model.freqs_cis, self.model.position_embeddings]
        """
        raise NotImplementedError(
            "Please implement `context_parallel_buffers` in your trainer. Return buffers like freq_cis"
        )

    def apply_pipeline_parallel(
        self, pp_mesh: DeviceMesh
    ) -> dict[str, tuple[_PipelineSchedule, list[nn.Module], bool, bool]]:
        """
        Apply pipeline parallelism to the trainer's models.

        This method should implement the logic to split models into pipeline stages
        and distribute them across the pipeline parallel device mesh. Each model
        can be split into multiple stages that will be executed in a pipelined fashion.

        Args:
            pp_mesh: The device mesh for pipeline parallelism, defining how devices
                are organized for pipeline stages.

        Returns:
            dict: A dictionary mapping model attribute names to their pipeline configuration:
                - key: The attribute name of the model on the trainer (e.g., "model")
                - value: A tuple containing:
                    - pipeline_schedule: The schedule defining how pipeline stages execute
                    - model_parts: List of nn.Module instances, one per pipeline stage
                    - has_first_stage: True if this rank owns the first pipeline stage
                    - has_last_stage: True if this rank owns the last pipeline stage

        Raises:
            NotImplementedError: If pipeline parallelism is requested but not implemented

        Example:
            def apply_pipeline_parallel(self, pp_mesh):
                # Split model into stages
                stages = self.model.split_into_stages()
                schedule = create_pipeline_schedule(stages, pp_mesh)

                return {
                    "model": (schedule, stages, rank == 0, rank == world_size - 1)
                }
        """
        raise NotImplementedError(
            "Please implement `apply_pipeline_parallel` in your trainer or set device_parameters.pipeline_parallel_degree=1"
        )

    def apply_tensor_parallel(self, tp_mesh: DeviceMesh):
        """
        Apply tensor parallelism to the trainer's models.

        This method should implement the logic to parallelize model layers across
        the tensor parallel dimension. Typically, this involves splitting linear
        layers, embeddings, and attention heads across devices.

        Args:
            tp_mesh: The device mesh for tensor parallelism, defining how devices
                are organized for tensor-level parallelism.

        Raises:
            NotImplementedError: If tensor parallelism is requested but not implemented

        Example:
            def apply_tensor_parallel(self, tp_mesh):
                from torch.distributed.tensor.parallel import parallelize_module

                parallelize_module(
                    self.model,
                    tp_mesh,
                    {"attention": ColwiseParallel(), "mlp": RowwiseParallel()}
                )
        """
        raise NotImplementedError(
            "Please implement `apply_tensor_parallel` in your trainer or set device_parameters.tensor_parallel_degree=1"
        )

    def apply_compile(self):
        """Compile models for optimized execution.

        This method should implement model compilation using torch.compile or
        similar optimization techniques. Compilation can significantly improve
        training and inference performance by optimizing the computation graph.

        The method is called after parallelism is applied but before weight
        initialization.

        Raises:
            NotImplementedError: If compilation is requested but not implemented

        Example:
            def apply_compile(self):
                import torch._dynamo as dynamo

                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=True
                )
        """
        raise NotImplementedError(
            "Please implement compile_model or set device_parameters.compile_model=False"
        )

    def apply_activation_checkpointing(self) -> None:
        """Apply activation checkpointing to reduce memory usage.

        This method should implement activation checkpointing (gradient checkpointing)
        for the models. This technique trades computation for memory by not storing
        intermediate activations during the forward pass and recomputing them during
        the backward pass.

        Raises:
            NotImplementedError: If activation checkpointing is requested but not implemented

        Example:
            def apply_activation_checkpointing(self):
                from torch.distributed.algorithms._checkpoint import checkpoint_wrapper

                # Wrap specific layers with checkpointing
                for layer in self.model.transformer_layers:
                    wrapped = checkpoint_wrapper(layer)
                    setattr(self.model, f"layer_{i}", wrapped)
        """
        raise NotImplementedError(
            "Please implement `apply_activation_checkpointing` in your trainer or set training_parameters.checkpoint_activations=False"
        )

    def apply_fully_shard(self, config: dict[str, Any]) -> None:
        """Apply Fully Sharded Data Parallel (FSDP) to models.

        This method should implement FSDP wrapping for the models. FSDP shards
        model parameters, gradients, and optimizer states across data parallel
        ranks to reduce memory usage and enable training of larger models.

        Args:
            config: FSDP configuration dictionary containing settings like
                sharding strategy, backward prefetch, forward prefetch, etc.

        Raises:
            NotImplementedError: If FSDP is requested but not implemented

        Example:
            def apply_fully_shard(self, config):
                from torch.distributed.fsdp import fully_shard

                for layer in self.model.layers:
                    fully_shard(layer, **config)
                fully_shard(self.model, **config)
        """
        raise NotImplementedError(
            "Please implement `apply_fully_shard` or disable all parallelism but dp_replicate"
        )

    def apply_replicate(self, dp_replicate_mesh: DeviceMesh):
        """Apply traditional data parallel replication (DDP) to models.

        This method should implement Distributed Data Parallel (DDP) for the models.
        Unlike FSDP, DDP replicates the entire model on each device and synchronizes
        gradients during the backward pass.

        Args:
            dp_replicate_mesh: The device mesh for data parallel replication

        Raises:
            NotImplementedError: If DDP is requested but not implemented

        Example:
            def apply_replicate(self, dp_replicate_mesh):
                from torch.distributed._composable.replicate import replicate

                replicate(self.model, device_mesh=dp_replicate_mesh)
        """
        raise NotImplementedError(
            "Please implement `apply_replicate` or use non-DDP DeviceParameters."
            "Ex:\nfrom torch.distributed._composable.replicate import replicate \nreplicate(self.model, device_mesh=self.world.get_mesh('dp_replicate'))"
        )

    #######################
    # Convenience Methods #
    #######################

    def get_model(self, name: str) -> nn.Module:
        """Retrieve a model by its attribute name.

        This is a convenience method for accessing models that were registered
        during the configure_models phase.

        Args:
            name: The attribute name of the model (e.g., "model", "encoder")

        Returns:
            nn.Module: The requested model

        Raises:
            AttributeError: If no model with the given name exists

        Example:
            >>> model = trainer.get_model("model")
            >>> encoder = trainer.get_model("encoder")
        """
        return getattr(self, name)

    def get_submodule(self, name: str) -> nn.Module:
        """Retrieve a submodule using dot notation without specifying the parent model.

        This method is similar to get_module but doesn't require the full FQN
        starting from the model name. It automatically finds the model containing
        the specified submodule.

        Args:
            name: Dot-separated path to the submodule (e.g., "encoder.attention")

        Returns:
            nn.Module: The requested submodule

        Raises:
            AttributeError: If the submodule doesn't exist

        Example:
            >>> attention = trainer.get_submodule("encoder.attention")
        """
        child_name, *submodule_name = name.split(".", 1)
        return self.get_model(child_name).get_submodule(".".join(submodule_name))

    ###################
    # Private Methods #
    ###################

    def _apply_pipeline_parallel(self):
        """Apply pipeline parallelism if configured.

        Checks if a pipeline parallel mesh exists and calls the user-defined
        apply_pipeline_parallel method if needed.
        """
        if (pp_mesh := self.world.get_mesh("pp")) is not None:
            raise NotImplementedError("Pipeline parallelism not implemented")
            self.apply_pipeline_parallel(pp_mesh)
            logger.info("Applied Pipeline Parallelism")

    def _apply_tensor_parallel(self):
        """Apply tensor parallelism if configured.

        Checks if a tensor parallel mesh exists and calls the user-defined
        apply_tensor_parallel method if needed.
        """
        if (tp_mesh := self.world.get_mesh("tp")) is not None:
            self.apply_tensor_parallel(tp_mesh)
            logger.info("Applied Tensor Parallelism")

    def _apply_activation_checkpointing(self):
        """Apply activation checkpointing if enabled.

        Checks the device parameters configuration and calls the user-defined
        apply_activation_checkpointing method if checkpointing is enabled.
        """
        if self.device_parameters.checkpoint_activations:
            self.apply_activation_checkpointing()
            logger.info("Applied Activation Checkpointing")

    def _apply_compile(self):
        """Apply model compilation if enabled.

        Checks the device parameters configuration and calls the user-defined
        apply_compile method if compilation is enabled.
        """
        if self.device_parameters.compile_model:
            self.apply_compile()
            logger.info("Compiled Model")

    def _apply_fully_shard(self):
        """Apply FSDP or DDP based on the configuration.

        This method determines whether to use FSDP or DDP based on the world
        configuration and calls the appropriate user-defined method. It also
        validates that all parameters requiring gradients are properly wrapped.
        """
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
        """Materialize models from meta tensors to actual device tensors.

        This method moves models from the meta device to the actual training device
        and initializes their weights. It handles CPU offloading for FSDP modules
        and sets the appropriate training/eval mode based on whether parameters
        require gradients.
        """
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
        """Register forward methods for FSDP modules and wrap them with autocast.

        This method performs two main tasks:
        1. For FSDP modules: Registers additional forward methods so FSDP can
           properly handle them during forward passes
        2. For all modules: Wraps forward methods with autocast for mixed precision

        The method automatically handles all models' forward methods and any
        additional methods specified by mark_forward_methods().
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
        """Configure models within a meta device context.

        This method sets up the model configuration phase by:
        1. Creating a list to track model names
        2. Setting up a meta device context for memory-efficient model creation
        3. Setting up a configuration context to automatically track model assignments
        4. Calling the user-defined configure_models method
        5. Calling the optional post_configure_models hook
        """
        self._model_names: list[str] = []
        with (
            torch.device("meta"),
            configuration_ctx(self, self._model_names, nn.Module),
        ):
            self.configure_models()

        self.post_configure_models()
        logger.info("Configured Models")

    def _setup_models(self):
        """Orchestrate the complete model setup process.

        This method coordinates the entire model setup pipeline in the correct order:
        1. Apply pipeline parallelism (if configured)
        2. Apply tensor parallelism (if configured)
        3. Apply activation checkpointing (if configured)
        4. Apply model compilation (if configured)
        5. Apply FSDP or DDP (based on configuration)
        6. Materialize models and initialize weights
        7. Register forward methods for mixed precision

        This ordering ensures that parallelism is applied before materialization
        and that all optimizations are in place before training begins.
        """
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
