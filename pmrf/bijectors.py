"""
Bijectors for transforming parameters.

These are re-exports from `parax <https://gvcallen.github.io/parax>`_, which in turn
takes them from the `distreqx <https://lockwo.github.io/distreqx>`_ library, filling in
any that the installed version of `distreqx` does not provide. The goal is to cover the
most common applications; for more advanced use-cases, simply use `distreqx` directly
instead.
"""
from parax.bijectors import (
    AbstractBijector as AbstractBijector,
    Chain as Chain,
    DiagLinear as DiagLinear,
    Exp as Exp,
    Identity as Identity,
    Inverse as Inverse,
    Leafwise as Leafwise,
    Permute as Permute,
    R2ToComplex as R2ToComplex,
    ScalarAffine as ScalarAffine,
    Shift as Shift,
    Sigmoid as Sigmoid,
    Softplus as Softplus,
    Tanh as Tanh,
    Transpose as Transpose,
    TriangularLinear as TriangularLinear,
)


__all__ = [
    'AbstractBijector',
    'Chain',
    'DiagLinear',
    'Exp',
    'Identity',
    'Inverse',
    'Leafwise',
    'Permute',
    'R2ToComplex',
    'ScalarAffine',
    'Shift',
    'Sigmoid',
    'Softplus',
    'Tanh',
    'Transpose',
    'TriangularLinear',
]
