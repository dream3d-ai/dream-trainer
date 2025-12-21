from pathlib import Path
from typing import Literal

from registry import Module
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from tqdm import tqdm

from dream_trainer.utils import logger

from .partial import LoadPartialCheckpointCallback


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
    ):
        super().__init__(checkpoint_path, resume_mode)
        if submodules and exclude_submodules:
            raise ValueError("submodules and exclude_submodules cannot both be set")

        self.output_path = Path(output_path)
        self.submodules = submodules
        self.exclude_submodules = exclude_submodules
        self.ignore_frozen_params = ignore_frozen_params
        self.overwrite = overwrite
        self.output_path.mkdir(parents=True, exist_ok=True)

    def pre_fit(self):
        super().pre_fit()

        with tqdm(self.trainer.named_models().items(), desc="Exporting models") as pbar:
            for name, model in pbar:
                if not isinstance(model, Module):
                    logger.warning(
                        f"Exporting only supports DreamTrainer Modules. Skipping {name}"
                    )
                    continue

                if self.submodules and name not in self.submodules:
                    continue
                elif self.exclude_submodules and name in self.exclude_submodules:
                    continue

                pbar.set_description(f"Exporting {name} to {self.output_path}")

                model.config.save_pretrained(
                    get_model_state_dict(
                        model,
                        options=StateDictOptions(
                            full_state_dict=True,
                            ignore_frozen_params=self.ignore_frozen_params,
                        ),
                    ),
                    str(self.output_path / name),
                    overwrite=self.overwrite,
                )

        exit()
