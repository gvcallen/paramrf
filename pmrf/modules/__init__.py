"""Parameter-aware wrappers for ParamRF modules."""

from pmrf.modules.base import Module as Module, is_module as is_module, validate as validate
from pmrf.modules.wrapped import (
    Probabilistic as Probabilistic,
    Tied as Tied,
)

__all__ = ["Module", "Probabilistic", "Tied", "is_module", "validate"]
