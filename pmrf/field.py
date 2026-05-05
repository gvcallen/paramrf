"""
ParamRF field specifier.
"""
import equinox as eqx

def field(*args, **kwargs):
    return eqx.field(*args, **kwargs)