"""
General constrants for the ParamRF library
"""
from typing import Callable, Literal, Union, Sequence, Any, Tuple, TypeVar
from numbers import Number
import jax.numpy as jnp

INF = 1e99
LOG_OF_NEG = -100

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

Evaluator = TypeVar('Evaluator')
Multioutput = TypeVar('Multioutput')
AbstractMinimiser = TypeVar('AbstractMinimiser')
AbstractSampler = TypeVar('AbstractSampler')
FeatureSpec = str | Callable | list[str | Callable]
ArrayFuncT = Callable[[jnp.ndarray], jnp.ndarray]
TreeAxisSpec = bool | Callable[[Any], bool]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray, Multioutput], jnp.ndarray]
Aggregation = Literal['raw_values', 'uniform_average', 'geometric_mean', 'convolution']
EvaluatorLike = str | list[str] | Evaluator | list[Evaluator]
Optimizer = AbstractMinimiser | Callable
Inferer = AbstractSampler

__all__ = []