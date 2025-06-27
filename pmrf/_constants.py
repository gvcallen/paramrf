from typing import Callable, Literal, Union, Sequence, Any, TypeVar, TYPE_CHECKING
from numbers import Number
import jax.numpy as jnp

from skrf.constants import INF, LOG_OF_NEG

NumberLike = Union[Number, Sequence[Number], jnp.ndarray]

FrequencyUnitT = Literal["Hz", "kHz", "MHz", "GHz", "THz"]

FREQ_UNITS: dict[FrequencyUnitT, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}

UNIT_DICT: dict[str] = {k.lower(): k for k in FREQ_UNITS}
MULTIPLIER_DICT = {k.lower(): v for k,v in FREQ_UNITS.items()}

PRIMARY_PROPERTIES = ('s', 'a')

FeatureT = tuple[str, tuple[int, int]]
ArrayFuncT = Callable[[jnp.ndarray], jnp.ndarray]
FeatureListT = list[FeatureT] | list[list[FeatureT]]
TreeAxisSpec = bool | Callable[[Any], bool]