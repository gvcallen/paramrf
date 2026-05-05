"""
ParamRF general field specifiers.
"""
from typing import Callable, Any

import equinox as eqx
import parax as prx

from pmrf.core import Model

def field(*args, **kwargs):
    return eqx.field(*args, **kwargs)

def frozen(*args, **kwargs):
    return prx.frozen(*args, **kwargs)

def model(factory: Callable[..., Model], *args, **kwargs) -> Any:
    """
    A field wrapper for initializing pmrf Models.

    This ensures that default models are instantiated freshly for every 
    parent object.

    Examples
    --------
    # Default initialization
    res: Resistor = pmrf.model(Resistor)

    # Initialization with custom default parameters
    clc: PiCLC = pmrf.model(PiCLC, C1=0.05e-12, L=0.1e-9)
    """
    # Create a deferred lambda that calls the factory with the provided args
    return eqx.field(default_factory=lambda: factory(*args, **kwargs))