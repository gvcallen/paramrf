from datetime import datetime
from typing import Union, Sequence
from numbers import Number

import pmrf.numpy as np

def time_string(format="%H:%M:%S"):
    return datetime.now().strftime(format)

NumberLike = Union[Number, Sequence[Number], np.ndarray]