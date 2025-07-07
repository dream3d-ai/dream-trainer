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
    """
    Filter that excludes specific modules from quantization by their fully qualified names.

    This filter ensures that:
    - Explicitly excluded modules are not quantized
    - Only nn.Linear modules are considered for quantization
    - Linear modules have dimensions divisible by 16 (required for float8 tensorcore)

    Attributes:
        exclude: Set of module names to exclude from quantization
    """

    def __init__(self, exclude: list[str]):
        """
        Initialize the filter with a list of module names to exclude.

        Args:
            exclude: List of fully qualified module names to exclude from quantization
        """
        self.exclude = set(exclude)

    def __call__(self, module: nn.Module, name: str) -> bool:
        """
        Determines whether a module should be quantized based on its name and type.

        Args:
            module: The module to check
            name: The fully qualified name of the module

        Returns:
            True if the module should be quantized, False otherwise

        Raises:
            ValueError: If the module is not nn.Linear or has dimensions not divisible by 16
        """
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
        """
        Validates that all excluded modules were encountered during filtering.

        Raises:
            AssertionError: If any excluded modules were not seen during filtering
        """
        assert len(self.exclude) == 0, (
            f"Not all excluded modules were seen. Missing: {self.exclude}"
        )


class ExcludeSubmodules(QuantizeModuleFilter):
    """
    Filter that excludes entire submodule trees from quantization based on module path prefixes.

    This filter allows excluding all modules under a specific path prefix. For example,
    excluding 'model.encoder' will exclude 'model.encoder.layer1', 'model.encoder.layer2', etc.

    Attributes:
        exclude: Set of module path prefixes to exclude
        _seen: Set of prefixes that were actually encountered during filtering
    """

    def __init__(self, exclude: list[str]):
        """
        Initialize the filter with a list of module path prefixes to exclude.

        Args:
            exclude: List of module path prefixes. Any module whose name starts with
                    these prefixes will be excluded from quantization.
        """
        self.exclude = set(exclude)
        self._seen = set()

    def __call__(self, module: nn.Module, name: str) -> bool:
        """
        Determines whether a module should be quantized based on its path prefix.

        Args:
            module: The module to check
            name: The fully qualified name of the module

        Returns:
            False if the module name matches or starts with any excluded prefix,
            True otherwise
        """
        for prefix in self.exclude:
            if name == prefix or name.startswith(prefix + "."):
                self._seen.add(prefix)
                return False
        return True

    def validate(self):
        """
        Validates that all excluded prefixes were encountered during filtering.

        Raises:
            AssertionError: If any excluded prefixes were not seen during filtering
        """
        missing = self.exclude - self._seen
        assert not missing, f"Not all excluded module prefixes were seen. Missing: {missing}"


class QuantizeConfigMixin(AbstractTrainerConfig):
    """
    Configuration mixin for quantization-related settings.

    This mixin can be combined with other trainer configurations to add
    quantization-specific configuration options.
    """

    ...


class QuantizeMixin(AbstractTrainer):
    """
    Mixin that adds quantization capabilities to a trainer.

    This mixin provides the infrastructure for quantizing models during training,
    including tracking which models have been quantized and defining module filters
    for selective quantization.

    Attributes:
        _quantized_models: List of model names that have been quantized
    """

    _quantized_models: list[str]

    def __init__(self, config: AbstractTrainerConfig):
        """
        Initialize the quantization mixin.

        Args:
            config: The trainer configuration
        """
        self._quantized_models = []
        super().__init__(config)

    def quantized_models(self) -> list[str]:
        """
        Get the list of models that have been quantized.

        Returns:
            List of model names that have been quantized
        """
        return self._quantized_models

    @abstractmethod
    def quantize_module_filters(self) -> dict[str, QuantizeModuleFilter]:
        """
        Define module filters for quantization.

        This method should return a dictionary mapping model names to their
        corresponding quantization filters. The filters determine which modules
        within each model should be quantized.

        Returns:
            Dictionary mapping model names to QuantizeModuleFilter instances
        """
        ...
