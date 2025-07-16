from typing import Callable

from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from dream_trainer import DreamTrainer
from dream_trainer.callbacks import Callback


class WeightTransferCallback(Callback[DreamTrainer]):
    """
    A callback that transfers weights from a source model to a target model after setup.

    This callback allows flexible weight transfer between models, supporting both direct transfers
    and transfers with transformation functions. The mapping dictionary specifies how parameters
    from the source model map to parameters in the target model.

    Args:
        source (str | Callable[[ProcessGroup | None], nn.Module]): Name of the source model to transfer weights
        from or a callable that constructs the source model itself
        target (str): Name of the target model to transfer weights to
        mapping (dict[str, None | str | tuple[str, Callable[[Tensor], Tensor]]]):
            A dictionary mapping target parameter names to source parameter specifications.
            If a target key is not found in the mapping, it is assumed to use the same parameter
            name in source.

            The values can be:
            - None: Do not transfer this parameter
            - str: Use the specified parameter name from source
            - tuple[str, Callable]: Use the specified parameter name and transform the tensor
              with the provided function before copying

    Example:
        ```python
        callback = WeightTransferCallback(
            source = "encoder",
            target = "decoder",
            mapping = {
                "layer1.weight": None, # layer1.weight is randomly initialized
                "layer2.weight": "encoder.layer2.weight",
                "layer3.weight": ("encoder.layer3.weight", lambda x: x.transpose(0, 1)),
                # All remaining parameters assumed to have the same name in the source
            },
        )
        ```
    """

    # TODO: Do this at the parameter loading stage (before materializing meta tensors) to reduce peak memory

    def __init__(
        self,
        source: str | Callable[[ProcessGroup | None], nn.Module],
        target: str,
        mapping: dict[str, None | str | tuple[str, Callable[[Tensor], Tensor]]],
        delete_source: bool = False,
    ):
        self.source = source
        self.target = target
        self.mapping = mapping
        self.delete_source = delete_source

    def post_setup(self):
        if isinstance(self.source, str):
            source = self.trainer.get_module(self.source)
        else:
            tp_mesh = self.trainer.world.get_mesh("tp")
            process_group = tp_mesh.get_group() if tp_mesh is not None else None
            source = self.source(process_group)

        source_state_dict = source.state_dict()
        target_state_dict = self.trainer.get_module(self.target).state_dict()

        source_keys, target_keys = set(source_state_dict.keys()), set(target_state_dict.keys())
        self.mapping.update({k: k for k in target_keys - self.mapping.keys()})
        mapping_keys = self.mapping.keys()
        mapping_values = {
            v if isinstance(v, str) else v[0] for v in self.mapping.values() if v is not None
        }

        assert target_keys == mapping_keys, (
            f"Target model keys must match mapping keys. \n\n"
            f"Found keys {mapping_keys - target_keys} in mapping but not in target model \n\n"
        )
        assert mapping_values.issubset(source_keys), (
            f"All mapping values (sources) must exist in source model or explicitly set to None. \n\n"
            f"Found keys {mapping_values - source_keys} in mapping values but not in source model"
        )

        for name, param in target_state_dict.items():
            if (replacement := self.mapping[name]) is None:
                continue

            if isinstance(replacement, str):
                source_param: Tensor = source_state_dict[replacement]
            else:
                source_param = replacement[1](source_state_dict[replacement[0]])

            assert param.shape == source_param.shape, (
                f"Parameter {name} in target model must have the same shape as the parameter in source model. \n\n"
                f"Found shape {param.shape} in target model but shape {source_param.shape} in source model{'.' if isinstance(replacement, str) else 'after applying mapping function.'}"
            )

            if isinstance((target_param := target_state_dict[name]), DTensor):
                source_param = DTensor.from_local(source_param, target_param.device_mesh)

            target_param.copy_(source_param)

        if self.delete_source:
            # Ensures the module no longer takes up memory (even if other objects store references)
            source.to_empty(device="meta")
            del source
