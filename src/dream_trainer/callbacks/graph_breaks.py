from typing import Any, Callable, override

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from ..trainer.base import BaseTrainer
from .callback import Callback

original_compile = torch.compile


class DynamoExplain(Exception):
    def __init__(self, explain):
        self.explain = explain


class TrainingStep(Exception):
    def __init__(self, batch, batch_idx):
        self.batch = batch
        self.batch_idx = batch_idx


class ExplainWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        explain = torch._dynamo.explain(self.module)(*args, **kwargs)  # type: ignore
        raise DynamoExplain(explain)

    def __getattr__(self, name: str):
        if name == "module":
            return super().__getattr__(name)
        return getattr(self.module, name)

    def __setattr__(self, name: str, value):
        if name == "module":
            return super().__setattr__(name, value)
        return setattr(self.module, name, value)


def explain_wrapper(func: Callable):
    def wrapper(*args, **kwargs):
        explain = torch._dynamo.explain(func)(*args, **kwargs)  # type: ignore
        raise DynamoExplain(explain)

    return wrapper


def get_name(module):
    if isinstance(module, nn.Module):
        return module.__class__.__name__

    if hasattr(module, "__self__"):
        # Handles Module.compile() calls where the captured forward function is compiled
        return f"{module.__self__.__class__.__name__}.{module.__name__}"

    return module.__name__


def please_find_graph_breaks(
    trainer: BaseTrainer,
    batch: dict[str, Any],
    batch_idx: int,
    path: str = "graph_breaks.log",
    fullgraph: bool = False,
):
    def _get_compiled(trainer) -> dict[str, nn.Module]:
        compiled = {}

        def _compile(module, *args, **kwargs):
            compiled[get_name(module)] = module
            return module

        torch.compile = _compile
        trainer.apply_compile()
        return compiled

    def _reset_compiled(trainer, name, module):
        def _compile(module_, *args, **kwargs):
            return module if get_name(module_) == name else module_

        torch.compile = _compile
        trainer.apply_compile()

    def _compile_specific(trainer, name):
        def _compile(module, *args, **kwargs):
            name = get_name(module)

            if name != name:
                return module

            if fullgraph:
                kwargs.pop("fullgraph", None)
                return original_compile(module, *args, **kwargs, fullgraph=True)

            if isinstance(module, nn.Module):
                return ExplainWrapper(module)

            return explain_wrapper(module)

        torch.compile = _compile
        trainer.apply_compile()
        return name

    # Count the number of compiled functions
    compiled = _get_compiled(trainer)
    explanations = []

    with tqdm(compiled.items(), desc="Finding graph breaks") as pbar:
        for name, module in pbar:
            pbar.set_description(f"Finding graph breaks in {name}")

            _compile_specific(trainer, name)

            try:
                with trainer.train_context():
                    trainer.training_step(batch, batch_idx)
            except DynamoExplain as exception:
                # Present only the most important information
                explain = exception.explain
                explain.ops_per_graph = None
                explain.out_guards = None
                explanations.append((name, explain))
            finally:
                _reset_compiled(trainer, name, module)
    if fullgraph:
        logger.success("Successfully compiled with fullgraph")
        return

    with open(path, "w") as f:
        for name, explain in explanations:
            f.write("=" * 100 + "\n")
            f.write(f"{name}\n")
            f.write(str(explain).replace(", line ", ":"))  # Formatting for vscode cmd+click
            f.write("\n\n")

    logger.success(f"Wrote graph breaks to {path}")


class FindGraphBreaksCallback(Callback[BaseTrainer]):
    """
    Find graph breaks in the trainer. This will check every compiled function in the trainer and
    write all the graph breaks to a file.

    NOTE: We only check for graph breaks in training steps.

    Args:
        log_file: File to write graph breaks to.
        skip: Number of steps to skip before finding graph breaks.
    """

    def __init__(
        self, log_file: str = "graph_breaks.log", skip: int = 0, fullgraph: bool = False
    ):
        self.log_file = log_file
        self.skip = skip
        self.full_graph = fullgraph

    @override
    def pre_launch(self):
        self.trainer.device_parameters.compile_model = False
        self.trainer.training_parameters.num_sanity_val_steps = 0

    @override
    def pre_train_step(self, batch: dict[str, Any], batch_idx: int):
        if batch_idx < self.skip:
            return

        please_find_graph_breaks(
            trainer=self.trainer,
            batch=batch,
            batch_idx=batch_idx,
            path=self.log_file,
            fullgraph=self.full_graph,
        )
        exit()
