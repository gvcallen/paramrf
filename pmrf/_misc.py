from datetime import datetime
from typing import Union, Sequence
from numbers import Number

import pmrf.numpy as np
from pmrf.numpy import USE_JAX
if USE_JAX:
    from equinox import field
else:
    from dataclasses import field

def time_string(format="%H:%M:%S"):
    return datetime.now().strftime(format)

NumberLike = Union[Number, Sequence[Number], np.ndarray]