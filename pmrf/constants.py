"""
General constrants for the ParamRF library
"""
from typing import TypeVar, Callable, Literal, Union, Sequence, Any, Tuple
from numbers import Number
import jax.numpy as jnp

from skrf.constants import INF, LOG_OF_NEG

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except:
    RANK = 0
    MPI_AVAILABLE = False
    
MIN_PERCENTILE = 0.01
MAX_PERCENTILE = 0.99

NumberLike = Union[Number, Sequence[Number], jnp.ndarray]
IndexArray = Union[int, slice, Sequence[int], jnp.ndarray, Tuple, None, type(Ellipsis)]

FrequencyUnitT = Literal["Hz", "kHz", "MHz", "GHz", "THz"]

FREQ_UNITS: dict[FrequencyUnitT, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}

UNIT_DICT: dict[str] = {k.lower(): k for k in FREQ_UNITS}
MULTIPLIER_DICT = {k.lower(): v for k,v in FREQ_UNITS.items()}

PRIMARY_PROPERTIES = ('s', 'a', 'y', 'z')

# Flat, structured feature type
FeatureT = tuple[str, str, tuple[int, int]]

ModelT = TypeVar('ModelT', bound='Model')
FeatureFunctionT = Callable[[ModelT | jnp.ndarray], jnp.ndarray]
ModelParametersT = Union[ModelT | jnp.ndarray]

# Alias/input feature types
FeatureSpecScalarT = str | tuple[str, str] | FeatureT
FeatureSpecSequenceT = Sequence[FeatureSpecScalarT]
FeatureSpecDictT = dict[str, FeatureSpecScalarT | FeatureSpecSequenceT]
FeatureSpecT = FeatureSpecScalarT | FeatureSpecSequenceT | FeatureSpecDictT | FeatureFunctionT


ArrayFuncT = Callable[[jnp.ndarray], jnp.ndarray]
TreeAxisSpec = bool | Callable[[Any], bool]

__all__ = []