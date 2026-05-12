"""
Undocumented constants.
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
    
NumberLike = Union[Number, Sequence[Number], jnp.ndarray]
IndexArray = Union[int, slice, Sequence[int], jnp.ndarray, Tuple, None, type(Ellipsis)]

FrequencyUnitT = Literal["Hz", "kHz", "MHz", "GHz", "THz"]

UNIT_TO_MULTIPLER: dict[FrequencyUnitT, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
UNIT_LOWER_TO_FORMATED: dict[str] = {k.lower(): k for k in UNIT_TO_MULTIPLER}
UNIT_LOWER_TO_MULTIPLER = {k.lower(): v for k,v in UNIT_TO_MULTIPLER.items()}

PRIMARY_PROPERTIES = ('s', 'a', 'y', 'z')

TreeAxisSpec = bool | Callable[[Any], bool]
AggregationKind = Literal['raw_values', 'uniform_average', 'geometric_mean', 'convolution']

__all__ = []