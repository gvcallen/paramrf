"""
ParamRF general field specifiers.
"""
from typing import Any
import equinox as eqx
import parax as prx

def field(*args, **kwargs):
    return eqx.field(*args, **kwargs)

def frozen(*args, **kwargs):
    return field(*args, converter=prx.as_frozen, **kwargs)