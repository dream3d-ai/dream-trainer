from abc import ABC, abstractmethod

import torch.nn as nn

from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig


class QuantizeModuleFilter(ABC):
    """
    Abstract class for filtering modules during quantization.
    """

    @abstractmethod
    def __call__(self, module: nn.Module, name: str) -> bool:
        """
        Determines whether a module should be quantized.

        Args:
            module: The module to check
            name: The fully qualified name of the module

        Returns:
            True if the module should be quantized, False otherwise
        """
        pass

    def validate(self):
        """
        Validates that the filter was applied correctly.
        Will be called after quantization is complete.
        """
        pass

    def __add__(self, other: "QuantizeModuleFilter") -> "QuantizeModuleFilter":
        """
        Returns a new QuantizeModuleFilter that applies both filters sequentially.
        The module will be quantized only if both filters return True.
        """

        class CombinedQuantizeModuleFilter(QuantizeModuleFilter):
            def __init__(self, filter1, filter2):
                self.filter1 = filter1
                self.filter2 = filter2

            def __call__(self, module: nn.Module, name: str) -> bool:
                return self.filter1(module, name) and self.filter2(module, name)

            def validate(self):
                self.filter1.validate()
                self.filter2.validate()

        return CombinedQuantizeModuleFilter(self, other)


class ExcludeModuleByName(QuantizeModuleFilter):
    def __init__(self, exclude: list[str]):
        self.exclude = set(exclude)

    def __call__(self, module: nn.Module, name: str) -> bool:
        if name in self.exclude:
            self.exclude.remove(name)
            return False

        if not isinstance(module, nn.Linear):
            raise ValueError(f"Exclusion list contains a non nn.Linear module: {name}")

        # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
        dims_multiples_of_16 = (
            module.weight.shape[0] % 16 == 0 and module.weight.shape[1] % 16 == 0
        )
        if not dims_multiples_of_16:
            raise ValueError(
                f"Linear layer {name} has in_features or out_features not divisible by 16. "
                "Please explicitly exclude this module from FP8 quantization."
            )
            return False

        return True

    def validate(self):
        assert len(self.exclude) == 0, (
            f"Not all excluded modules were seen. Missing: {self.exclude}"
        )


class ExcludeSubmodules(QuantizeModuleFilter):
    def __init__(self, exclude: list[str]):
        self.exclude = set(exclude)
        self._seen = set()

    def __call__(self, module: nn.Module, name: str) -> bool:
        for prefix in self.exclude:
            if name == prefix or name.startswith(prefix + "."):
                self._seen.add(prefix)
                return False
        return True

    def validate(self):
        missing = self.exclude - self._seen
        assert not missing, f"Not all excluded module prefixes were seen. Missing: {missing}"


class QuantizeConfigMixin(AbstractTrainerConfig): ...


class QuantizeMixin(AbstractTrainer):
    _quantized_models: list[str]

    def __init__(self, config: AbstractTrainerConfig):
        self._quantized_models = []
        super().__init__(config)

    def quantized_models(self) -> list[str]:
        return self._quantized_models

    @abstractmethod
    def quantize_module_filters(self) -> dict[str, QuantizeModuleFilter]: ...
