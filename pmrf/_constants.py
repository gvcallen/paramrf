from typing import TypeVar, Callable, Literal, Union, Sequence, Any, Tuple
from numbers import Number
import jax.numpy as jnp

from skrf.constants import INF, LOG_OF_NEG

NumberLike = Union[Number, Sequence[Number], jnp.ndarray]
IndexArray = Union[int, slice, Sequence[int], jnp.ndarray, Tuple, None, type(Ellipsis)]

FrequencyUnitT = Literal["Hz", "kHz", "MHz", "GHz", "THz"]

FREQ_UNITS: dict[FrequencyUnitT, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}

UNIT_DICT: dict[str] = {k.lower(): k for k in FREQ_UNITS}
MULTIPLIER_DICT = {k.lower(): v for k,v in FREQ_UNITS.items()}

PRIMARY_PROPERTIES = ('s', 'a')

# Flat, structured feature type
FeatureT = tuple[str, str, tuple[int, int]]

# Alias/input feature types
FeatureInputScalarT = str | tuple[str, str] | FeatureT
FeatureInputSequenceT = Sequence[FeatureInputScalarT]
FeatureInputDictT = dict[str, FeatureInputScalarT | FeatureInputSequenceT]
FeatureInputT = FeatureInputScalarT | FeatureInputSequenceT | FeatureInputDictT

ModelT = TypeVar('ModelT', bound='Model')
FeatureFunctionT = Callable[[ModelT | jnp.ndarray], jnp.ndarray]
ModelParametersT = Union[ModelT | jnp.ndarray]

ArrayFuncT = Callable[[jnp.ndarray], jnp.ndarray]
TreeAxisSpec = bool | Callable[[Any], bool]