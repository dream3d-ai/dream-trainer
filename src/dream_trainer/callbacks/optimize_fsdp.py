from typing import TYPE_CHECKING, Literal, cast, override

from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.utils.hooks import RemovableHandle

from dream_trainer.trainer import DreamTrainer
from dream_trainer.utils import logger

from .callback import Callback

if TYPE_CHECKING:
    from rich.tree import Tree as Tree


class _Node:
    """
    Class for storing and printing a tree of FSDP modules.
    """

    def __init__(
        self, name: str, origin: Literal["pre_forward_call", "post_forward_call", "prefetch"]
    ):
        self.name = name
        self.origin = origin
        self.children: list[_Node] = []

    def add(
        self,
        name: str,
        origin: Literal["pre_forward_call", "post_forward_call", "prefetch"],
    ):
        self.children.append(_Node(name, origin))

    def _print(self):
        from rich.tree import Tree

        tree = Tree(self.name)
        for child in self.children:
            if child.origin == "prefetch":
                tree.add(f"[blue]Prefetched {child.name}")
            else:
                tree.add(child._print())

        return tree

    def __str__(self):
        return str(self._print())

    def __repr__(self):
        return f"Node(name={self.name}, origin={self.origin}, children={self.children})"

    def print(self):
        # Construct a rich tree
        from rich import print as rprint

        rprint("\n\n")
        rprint("Optimized FSDP Call/Prefetch Stack")
        rprint(self._print())
        rprint("\n")

    @classmethod
    def from_stack(
        cls,
        stack: list[
            tuple[str, bool, Literal["pre_forward_call", "post_forward_call", "prefetch"]]
        ],
    ) -> "_Node":
        root = _Node("root", "pre_forward_call")
        _stack = [root]

        for name, _, origin in stack:
            if origin == "prefetch":
                _stack[-1].add(name, origin)
                continue

            if _stack and _stack[-1].name == name:
                _stack.pop()
                continue

            _stack[-1].add(name, origin)
            _stack.append(_stack[-1].children[-1])

        return root


_original_prefetch_unshard = FSDPParamGroup._prefetch_unshard


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

    def __init__(self, prefetch: int = 1, display: bool = False):
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
        self.display = display

        self.stack: list[
            tuple[str, bool, Literal["pre_forward_call", "post_forward_call", "prefetch"]]
        ] = []
        self.hooks: list[RemovableHandle] = []

    def append_call(
        self,
        name: str,
        requires_grad: bool,
        origin: Literal["pre_forward_call", "post_forward_call", "prefetch"],
    ):
        """Create a hook function that records FSDP module execution order.

        This method returns a hook function that, when called, appends the module
        name and gradient requirement to the execution stack. This is used to
        trace the order in which FSDP modules are executed during forward pass.

        Args:
            name: Fully qualified name of the FSDP module.
            requires_grad: Whether the module has parameters that require gradients.
            *args: Additional positional arguments (unused, for hook compatibility).
            **kwargs: Additional keyword arguments (unused, for hook compatibility).

        Returns:
            Hook function that records module execution when called.
        """

        def hook(*args, **kwargs):
            self.stack.append((name, requires_grad, origin))

        return hook

    def append_prefetch(self):
        """Create a hook function that records FSDP module prefetch operations.

        This method returns a hook function that replaces the default FSDP prefetch
        behavior. When called, it records the prefetch operation in the execution
        stack before delegating to the original prefetch implementation.

        The hook tracks:
        - The fully qualified name of the FSDP module being prefetched
        - That the module does not require gradients (False)
        - That this is a prefetch operation ("prefetch")

        Returns:
            Hook function that records prefetch operations when called.
        """

        @staticmethod
        def _prefetch_unshard(
            target_fsdp_param_group: "FSDPParamGroup", pass_type: str
        ) -> None:
            fqn = target_fsdp_param_group._module_fqn or ", ".join(
                module.__class__.__name__ for module in target_fsdp_param_group.modules
            )
            self.stack.append((fqn, False, "prefetch"))
            _original_prefetch_unshard(target_fsdp_param_group, pass_type)

        return _prefetch_unshard

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

        for name, model in self.trainer.named_models().items():
            for module_name, module in model.named_modules():
                if isinstance(module, FSDPModule):
                    module = cast(nn.Module, module)
                    fqn = f"{name}.{module_name}" if module_name else name

                    self.hooks.extend(
                        [
                            module.register_forward_pre_hook(
                                self.append_call(
                                    fqn,
                                    any(p.requires_grad for p in module.parameters()),
                                    "pre_forward_call",
                                )
                            ),
                            module.register_forward_hook(
                                self.append_call(
                                    fqn,
                                    any(p.requires_grad for p in module.parameters()),
                                    "post_forward_call",
                                )
                            ),
                        ]
                    )

        logger.info(f"Found {len(self.hooks) // 2} forward and backward calls for FSDP modules")

    @override
    def pre_train_step(self, *_):
        """Unshard FSDP models asynchronously before training step.

        This method is called before each training step. It triggers asynchronous unsharding of
        the first all-gather of all FSDP model, allowing the unsharding operation to overlap
        with other computations and reducing the time spent waiting for data movement.

        Args:
            *_: Unused arguments from the trainer callback interface.
        """
        for _, model in self.trainer.named_models().items():
            if isinstance(model, FSDPModule):
                model.unshard(async_op=True)

    @override
    def post_train_step(self, _, batch_idx: int):
        """Set up prefetching based on traced execution order and clean up hooks.

        This method is called after the first training step. It uses the execution
        order recorded by the hooks to set up optimal prefetching for both forward
        and backward passes. After setting up prefetching, it removes all hooks
        and clears the execution stack since tracing is only needed once.

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

        # Add forward prefetching after first training step
        if self.trainer.local_batches == 0:
            prefetch_mode = (
                "conservative (singleton lists)"
                if self.prefetch == 1
                else "aggressive (multi-module lists)"
            )
            logger.info(
                f"Setting up {prefetch_mode} prefetch for {len(self.stack)} forward and backward calls "
                f"with prefetch factor {self.prefetch}"
            )

            # Get the modules in order of execution
            ordered_forward_modules = cast(
                list[FSDPModule],
                [
                    self.trainer.get_module(fqn)
                    for fqn, _, origin in self.stack
                    if origin == "pre_forward_call"
                ],
            )
            ordered_backwards_modules = cast(
                list[FSDPModule],
                [
                    self.trainer.get_module(fqn)
                    for fqn, requires_grad, origin in self.stack[::-1]
                    if requires_grad and origin == "pre_forward_call"
                ],
            )

            # Set up prefetching
            for i, module in enumerate(ordered_forward_modules):
                if i == 0:
                    module.set_modules_to_forward_prefetch(
                        ordered_forward_modules[1 : self.prefetch]
                    )
                else:
                    module.set_modules_to_forward_prefetch(
                        ordered_forward_modules[i + self.prefetch : i + 1 + self.prefetch]
                    )

            for i, module in enumerate(ordered_backwards_modules):
                if i == 0:
                    module.set_modules_to_backward_prefetch(
                        ordered_backwards_modules[1 : self.prefetch]
                    )
                else:
                    module.set_modules_to_backward_prefetch(
                        ordered_backwards_modules[i + self.prefetch : i + 1 + self.prefetch]
                    )

            # Clear the stack for second training step
            self.stack.clear()

            # Add hook to log prefetching
            FSDPParamGroup._prefetch_unshard = self.append_prefetch()

        # Log a tree inorder of forward calls and prefetching
        elif self.trainer.local_batches == 1:
            FSDPParamGroup._prefetch_unshard = staticmethod(_original_prefetch_unshard)

            for hook in self.hooks:
                hook.remove()

            if self.display:
                _Node.from_stack(self.stack).print()
