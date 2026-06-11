from pathlib import Path
from typing import Any, Literal

from registry import Module
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from tqdm import tqdm

from dream_trainer.utils import logger

from .partial import LoadPartialCheckpointCallback
from .s3 import is_s3_uri, join_s3_uri


class ExportCallback(LoadPartialCheckpointCallback):
    def __init__(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path,
        submodules: list[str] | None = None,
        exclude_submodules: list[str] | None = None,
        ignore_frozen_params: bool = False,
        overwrite: bool = False,
        resume_mode: Literal["min", "max", "last"] | int = "last",
        storage_options: dict[str, Any] | None = None,
    ):
        super().__init__(checkpoint_path, resume_mode, storage_options=storage_options)
        if submodules and exclude_submodules:
            raise ValueError("submodules and exclude_submodules cannot both be set")

        self.output_path: str | Path = (
            str(output_path).rstrip("/") if is_s3_uri(output_path) else Path(output_path)
        )
        self.submodules = submodules
        self.exclude_submodules = exclude_submodules
        self.ignore_frozen_params = ignore_frozen_params
        self.overwrite = overwrite
        if not is_s3_uri(self.output_path):
            assert isinstance(self.output_path, Path)
            self.output_path.mkdir(parents=True, exist_ok=True)

    def _resolve_export_targets(self) -> list[tuple[str, Module]]:
        """Resolve submodule paths (e.g. 'pipe.model') to (name, Module) pairs."""
        if self.submodules:
            targets = []
            for fqn in self.submodules:
                model = self.trainer.get_module(fqn)
                if not isinstance(model, Module):
                    logger.warning(
                        f"Exporting only supports DreamTrainer Modules. Skipping {fqn}"
                    )
                    continue
                targets.append((fqn, model))
            return targets

        return [
            (name, model)
            for name, model in self.trainer.named_models().items()
            if isinstance(model, Module)
            and not (self.exclude_submodules and name in self.exclude_submodules)
        ]

    def pre_fit(self):
        super().pre_fit()

        targets = self._resolve_export_targets()
        with tqdm(targets, desc="Exporting models") as pbar:
            for name, model in pbar:
                pbar.set_description(f"Exporting {name} to {self.output_path}")

                model.config.save_pretrained(
                    get_model_state_dict(
                        model,
                        options=StateDictOptions(
                            full_state_dict=True,
                            ignore_frozen_params=self.ignore_frozen_params,
                        ),
                    ),
                    (
                        join_s3_uri(self.output_path, name)
                        if is_s3_uri(self.output_path)
                        else str(self.output_path / name)
                    ),
                    overwrite=self.overwrite,
                    storage_options=self.storage_options,
                )

        exit()
