try:
    import torchao  #  noqa: F401
except ImportError:
    raise ImportError(
        "torchao is not installed. Please install it with `pip install dream-trainer[fp8]` to use the Fp8Quantization callback."
    )

from .callback import Fp8Quantization
from .types import Fp8QuantizeConfig

__all__ = ["Fp8Quantization", "Fp8QuantizeConfig"]
