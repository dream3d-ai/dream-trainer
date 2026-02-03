import warnings
from typing import TYPE_CHECKING, cast, override

from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.utils.hooks import RemovableHandle

from dream_trainer.trainer import DreamTrainer
from dream_trainer.utils import logger

from .callback import Callback

if TYPE_CHECKING:
    from rich.tree import Tree as Tree


class OptimizeFSDP(Callback[DreamTrainer]):
    """FSDP optimization callback that improves training performance through intelligent prefetching.

    This callback optimizes Fully Sharded Data Parallel (FSDP) training by:
    1. Tracing the execution order of FSDP modules during the first training step
    2. Setting up prefetching for both forward and backward passes based on the traced order
    3. Unsharding models asynchronously before each training step

    The prefetching mechanism overlaps data movement with computation, reducing idle time
    and improving overall training throughput.

    Prefetch Behavior:
    - prefetch=1: Uses singleton lists, providing the same all-gather overlap as
      default behavior but issues prefetched all-gathers earlier from the CPU
    - prefetch>=2: Enables more aggressive overlap with higher memory usage due
      to additional reserved memory for prefetched modules

    Args:
        prefetch: Number of modules to prefetch ahead. Higher values increase
            memory usage but may improve performance. Must be >= 1. Defaults to 1.
        display: Whether to display the tree of FSDP modules after construction. Defaults to True.

    Attributes:
        prefetch: The number of modules to prefetch ahead.
        stack: List of (module_name, requires_grad) tuples tracking execution order.
        hooks: List of registered forward hooks for tracing module execution.
    """

    def __init__(
        self,
        prefetch: int = 1,
        allocate_memory_from_process_group_for_comm: bool = False,
        display: bool = False,
        unshard_on_step: bool = True,
    ):
        """Initialize the FSDP optimization callback.

        Args:
            prefetch: Number of modules to prefetch ahead. Must be >= 1.
                Values >= 2 enable more aggressive overlap but use more memory.

        Raises:
            ValueError: If prefetch is less than 1.
        """
        if prefetch < 1:
            raise ValueError(f"prefetch must be >= 1, got {prefetch}")

        self.prefetch = prefetch
        self.unshard_on_step = unshard_on_step
        self.allocate_memory_from_process_group_for_comm = (
            allocate_memory_from_process_group_for_comm
        )
        self.display = display

        # Track execution order per model (prefetching only valid within a model)
        self.forward_stacks: dict[str, list[str]] = {}
        self.backward_stacks: dict[str, list[str]] = {}

        # Track modules that haven't been called yet, per model
        self.pending_forward_modules: dict[str, set[str]] = {}
        self.pending_backward_modules: dict[str, set[str]] = {}

        self.forward_hooks: list[RemovableHandle] = []
        self.backward_hooks: list[RemovableHandle] = []

    def append_forward_call(self, model_name: str, fqn: str):
        """Create a hook function that records FSDP module execution order.

        This method returns a hook function that, when called, appends the module
        name to the execution stack for its model. This is used to trace the order
        in which FSDP modules are executed during forward pass.

        Args:
            model_name: Name of the top-level model this module belongs to.
            fqn: Fully qualified name of the FSDP module.

        Returns:
            Hook function that records module execution when called.
        """

        def hook(*args, **kwargs):
            self.forward_stacks[model_name].append(fqn)
            self.pending_forward_modules[model_name].discard(fqn)

        return hook

    def append_backward_call(self, model_name: str, fqn: str):
        """Create a hook function that records FSDP module execution order.

        This method returns a hook function that, when called, appends the module
        name to the execution stack for its model. This is used to trace the order
        in which FSDP modules are executed during backward pass.

        Args:
            model_name: Name of the top-level model this module belongs to.
            fqn: Fully qualified name of the FSDP module.

        Returns:
            Hook function that records module execution when called.
        """

        def hook(*args, **kwargs):
            self.backward_stacks[model_name].append(fqn)
            self.pending_backward_modules[model_name].discard(fqn)

        return hook

    @override
    def post_setup(self):
        """Set up forward hooks to trace FSDP module execution order.

        This method is called before training begins. It registers forward pre-hooks
        on all FSDP modules to trace their execution order during the first training
        step. This information is later used to set up optimal prefetching patterns.

        The hooks record both the module name and whether it has parameters requiring
        gradients, which is needed to determine backward pass prefetching.
        """
        logger.info("Tracing forward and backward calls for FSDP modules")

        # Suppress the "Full backward hook is firing when gradients are computed with
        # respect to module outputs since no inputs require gradients" warning.
        warnings.filterwarnings(
            "ignore",
            message="Full backward hook is firing",
            category=UserWarning,
        )

        for name, model in self.trainer.named_models().items():
            # Initialize per-model tracking structures
            self.forward_stacks[name] = []
            self.backward_stacks[name] = []
            self.pending_forward_modules[name] = set()
            self.pending_backward_modules[name] = set()

            for module_name, module in model.named_modules():
                if isinstance(module, FSDPModule):
                    module.set_allocate_memory_from_process_group_for_comm(
                        self.allocate_memory_from_process_group_for_comm
                    )

                    if self.prefetch > 1:
                        module._set_unshard_async_op(True)

                        module = cast(nn.Module, module)
                        fqn = f"{name}.{module_name}" if module_name else name

                        self.pending_forward_modules[name].add(fqn)
                        self.forward_hooks.extend(
                            [
                                module.register_forward_pre_hook(
                                    self.append_forward_call(name, fqn)
                                ),
                            ]
                        )

                        # Only register backward hooks on modules with trainable parameters
                        if any(p.requires_grad for p in module.parameters()):
                            self.pending_backward_modules[name].add(fqn)
                            self.backward_hooks.extend(
                                [
                                    module.register_full_backward_pre_hook(
                                        self.append_backward_call(name, fqn)
                                    ),
                                ]
                            )
        logger.info(
            f"Found {len(self.forward_hooks)} forward and {len(self.backward_hooks)} backward calls for FSDP modules"
        )

    @override
    def pre_train_step(self, *_):
        """Unshard FSDP models asynchronously before training step.

        This method is called before each training step. It triggers asynchronous unsharding of
        the first all-gather of all FSDP model, allowing the unsharding operation to overlap
        with other computations and reducing the time spent waiting for data movement.

        Args:
            *_: Unused arguments from the trainer callback interface.
        """
        if not self.unshard_on_step:
            return

        for _, model in self.trainer.named_models().items():
            if isinstance(model, FSDPModule):
                model.unshard(async_op=True)

    @override
    def pre_validation_step(self, *_):
        """Unshard FSDP models asynchronously before validation step.

        This method is called before each validation step. It triggers asynchronous unsharding of
        the first all-gather of all FSDP model, allowing the unsharding operation to overlap
        with other computations and reducing the time spent waiting for data movement.

        Args:
            *_: Unused arguments from the trainer callback interface.
        """
        if not self.unshard_on_step:
            return

        for _, model in self.trainer.named_models().items():
            if isinstance(model, FSDPModule):
                model.unshard(async_op=True)

    def _setup_prefetch_for_model(self, model_name: str):
        """Set up prefetching for a single model that has completed tracing.

        Args:
            model_name: Name of the model to set up prefetching for.
        """
        forward_stack = self.forward_stacks[model_name]
        backward_stack = self.backward_stacks[model_name]

        if not forward_stack:
            return

        prefetch_mode = (
            "conservative (singleton lists)"
            if self.prefetch == 1
            else "aggressive (multi-module lists)"
        )
        logger.info(
            f"Setting up {prefetch_mode} prefetch for model '{model_name}': "
            f"{len(forward_stack)} forward calls and {len(backward_stack)} backward calls "
            f"with prefetch factor {self.prefetch}"
        )

        # Get the modules in order of execution
        ordered_forward_modules = cast(
            list[FSDPModule],
            [self.trainer.get_module(fqn) for fqn in forward_stack],
        )
        ordered_backwards_modules = cast(
            list[FSDPModule],
            [self.trainer.get_module(fqn) for fqn in backward_stack],
        )

        # Set up prefetching without overlap:
        # - Module 0 primes the pipeline by prefetching `prefetch` modules
        # - Subsequent modules each prefetch one new module to maintain the window
        for i, module in enumerate(ordered_forward_modules):
            if i == 0:
                module.set_modules_to_forward_prefetch(
                    ordered_forward_modules[1 : 1 + self.prefetch]
                )
            else:
                module.set_modules_to_forward_prefetch(
                    ordered_forward_modules[i + self.prefetch : i + self.prefetch + 1]
                )

        for i, module in enumerate(ordered_backwards_modules):
            if i == 0:
                module.set_modules_to_backward_prefetch(
                    ordered_backwards_modules[1 : 1 + self.prefetch]
                )
            else:
                module.set_modules_to_backward_prefetch(
                    ordered_backwards_modules[i + self.prefetch : i + self.prefetch + 1]
                )

    @override
    def post_train_step(self, _, batch_idx: int):
        """Set up prefetching based on traced execution order and clean up hooks.

        This method is called after each training step. It processes each model
        independently as soon as all its FSDP modules have been called at least once,
        rather than waiting for all models to complete tracing.

        The prefetching setup works by:
        1. Using the forward execution order for forward prefetching
        2. Using the reverse order (filtering only modules with gradients) for backward prefetching
        3. Setting each module to prefetch the next `prefetch` modules in sequence

        Prefetch list behavior:
        - Single module lists (prefetch=1): Same overlap as default, earlier CPU scheduling
        - Multi-module lists (prefetch>=2): More aggressive overlap, higher memory usage

        Args:
            *_: Unused arguments from the trainer callback interface.
        """
        # Process each model independently as soon as it's ready
        ready_models = [
            model_name
            for model_name in self.forward_stacks
            if not self.pending_forward_modules.get(model_name)
            and not self.pending_backward_modules.get(model_name)
        ]

        for model_name in ready_models:
            self._setup_prefetch_for_model(model_name)

            # Clean up this model's tracking data
            del self.forward_stacks[model_name]
            del self.backward_stacks[model_name]
            self.pending_forward_modules.pop(model_name, None)
            self.pending_backward_modules.pop(model_name, None)

        # Log progress for models still pending
        if self.forward_stacks:
            pending_info = [
                f"{name}: {len(self.pending_forward_modules.get(name, set()))}f/{len(self.pending_backward_modules.get(name, set()))}b"
                for name in self.forward_stacks
            ]
            logger.debug(f"Waiting for FSDP modules: {', '.join(pending_info)}")
            return

        # All models have been processed, clean up hooks and remove callback
        for hook in self.forward_hooks:
            hook.remove()
        self.forward_hooks.clear()

        for hook in self.backward_hooks:
            hook.remove()
        self.backward_hooks.clear()

        self.pop_self()
