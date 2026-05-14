"""
Constraints for parameters.

Can be used for parameter factories in :mod:`pmrf.parameters`.

Builds on top of the library `Parax <https://gvcallen.github.io/parax>`_.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from parax.constraints import (
    AbstractConstraint as AbstractConstraint,
    RealLine as RealLine,
    Positive as Positive,
    Negative as Negative,
    Interval as Interval,
    GreaterThan as GreaterThan,
    LessThan as LessThan,
    intersect as intersect,
)
    
    
__all__ = [
    'AbstractConstraint',
    'RealLine',
    'Positive',
    'Negative',
    'Interval',
    'GreaterThan',
    'LessThan',
    'intersect',
]