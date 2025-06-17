from typing import Union, Sequence
from jaxtyping import Array, Float
import pmrf._numpy as np

from jaxtyping import Float, Array
from numbers import Number
from typing import Sequence, Union

Scalar = Float[Array, ""]
Vector = Float[Array, "dim"]
NumberLike = Union[Number, Sequence[Number], np.ndarray]