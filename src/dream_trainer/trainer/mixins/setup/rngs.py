from dataclasses import dataclass

import torch
from typing_extensions import override

from dream_trainer.trainer.abstract import AbstractTrainer, AbstractTrainerConfig
from dream_trainer.utils import logger
from dream_trainer.utils.common import configuration_ctx


@dataclass(kw_only=True)
class RNGSetupConfigMixin(AbstractTrainerConfig):
    """Configuration mixin for RNG setup functionality.

    This class serves as a base configuration for trainers that need named
    random number generator setup capabilities. It inherits from
    AbstractTrainerConfig and can be extended with RNG-specific parameters.
    """

    ...


class RNGSetupMixin(AbstractTrainer):
    """Mixin that handles configuration and tracking of random number generators.

    This mixin provides a framework for registering `torch.Generator` instances
    on the trainer so they can be referenced by name, checkpointed, and restored
    on resume — mirroring how models, optimizers, and schedulers are tracked.

    Generators assigned as attributes during configure_rngs() are collected
    automatically and exposed via named_rngs(). BaseTrainer includes their
    states in the trainer state dict so sampling streams resume exactly.

    Attributes:
        config (RNGSetupConfigMixin): Configuration for RNG setup
        _rng_names (list[str]): Names of generator attributes registered during
            configuration

    Example:
        class MyTrainer(RNGSetupMixin):
            def configure_rngs(self):
                self.val_noise_rng = torch.Generator(device="cpu").manual_seed(self.seed)
    """

    config: RNGSetupConfigMixin

    ###########################
    # AbstractTrainer Methods #
    ###########################

    @override
    def named_rngs(self) -> dict[str, torch.Generator]:
        """Return a dictionary mapping RNG names to their generator instances.

        This method provides access to all generators registered during the
        configure_rngs phase. Generator names are collected automatically when
        generators are assigned as attributes during configuration.

        Returns:
            dict[str, torch.Generator]: Dictionary where keys are generator
                attribute names and values are the corresponding torch.Generator
                instances.

        Example:
            >>> trainer.named_rngs()
            {'val_noise_rng': <torch._C.Generator object>}
        """
        return {name: getattr(self, name) for name in self._rng_names}

    ########################
    # User-Defined Methods #
    ########################

    def configure_rngs(self):
        """Configure and instantiate random number generators used by the trainer.

        This optional method can be overridden to define `torch.Generator`
        instances. Generators should be assigned as attributes to the trainer
        instance; they are tracked automatically and included in checkpoints.

        The method is called at the start of the setup phase, so generators are
        available to init_weights(), configure_dataloaders(), and training steps.

        Example:
            def configure_rngs(self):
                self.val_noise_rng = torch.Generator(device="cpu").manual_seed(self.seed)
                self.sampling_rng = torch.Generator(device=self.world.device_type)
        """
        pass

    #######################
    # Convenience Methods #
    #######################

    def get_rng(self, name: str) -> torch.Generator:
        """Retrieve a generator by its attribute name.

        Args:
            name: The attribute name of the generator (e.g., "val_noise_rng")

        Returns:
            torch.Generator: The requested generator

        Raises:
            AttributeError: If no generator with the given name exists
        """
        return getattr(self, name)

    ###################
    # Private Methods #
    ###################

    def _setup_rngs(self):
        """Internal method to configure generators with tracking.

        This method wraps the user's configure_rngs() call in a context that
        automatically tracks generator instances as they're assigned.
        """
        self._rng_names: list[str] = []
        with configuration_ctx(self, self._rng_names, torch.Generator):
            self.configure_rngs()

        logger.info("Setup RNGs")
