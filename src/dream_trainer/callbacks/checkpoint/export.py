from pathlib import Path
from typing import Literal

from registry import Module
from tqdm import tqdm

from dream_trainer.utils import logger

from .partial import LoadPartialCheckpointCallback


class ExportCallback(LoadPartialCheckpointCallback):
    def __init__(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path,
        resume_mode: Literal["min", "max", "last"] = "last",
    ):
        super().__init__(checkpoint_path, resume_mode)
        self.output_path = Path(output_path)

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

                pbar.set_description(f"Exporting {name} to {self.output_path}")
                model.config.save_pretrained(model.state_dict(), str(self.output_path / name))

        exit()
