from datetime import datetime
from typing import Union, Sequence, Callable
from numbers import Number
from dataclasses import Field

import pmrf.numpy as np
from pmrf.numpy import USE_JAX
if USE_JAX:
    from equinox import field as base_field
else:
    from dataclasses import field as base_field    
    
def field(
    *,
    derived: bool = False,
    **kwargs,
):
    metadata = dict(kwargs.pop('metadata', {}))
    init = bool(kwargs.pop('init', (not derived)))
    if 'derived' in metadata:
        raise Exception("Cannot use metadata with `derived` already set.")
    metadata['derived'] = derived
    
    return base_field(init=init, metadata=metadata, **kwargs)

def time_string(format="%H:%M:%S"):
    return datetime.now().strftime(format)

NumberLike = Union[Number, Sequence[Number], np.ndarray]
