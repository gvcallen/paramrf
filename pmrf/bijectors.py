"""
Bijectors for transforming parameters.

These are re-exports from the `distreqx <https://lockwo.github.io/distreqx>`_ library,
with the goal of the covering the most common applications. For more advanced use-cases,
simply use `distreqx` directly instead.
"""
from distreqx.bijectors import (
    AbstractBijector as AbstractBijector,
    Chain as Chain,
    DiagLinear as DiagLinear,
    ScalarAffine as ScalarAffine,
    Sigmoid as Sigmoid,
    Softplus as Softplus,
    Tanh as Tanh,
    TriangularLinear as TriangularLinear,
)


__all__ = [
    'AbstractBijector',
    'Chain',
    'DiagLinear',
    'ScalarAffine',
    'Sigmoid',
    'Softplus',
    'Tanh',
    'TriangularLinear',
]

try:
    from distreqx.bijectors import Exp as Exp
    __all__.append('Exp')
except:
    pass

try:
    from distreqx.bijectors import Identity as Identity
    __all__.append('Identity')
except:
    pass

try:
    from distreqx.bijectors import Inverse as Inverse
    __all__.append('Inverse')
except:
    pass

try:
    from distreqx.bijectors import Leafwise as Leafwise
    __all__.append('Leafwise')
except:
    pass

try:
    from distreqx.bijectors import Scale as Scale
    __all__.append('Scale')
except:
    pass

try:
    from distreqx.bijectors import Shift as Shift
    __all__.append('Shift')
except:
    pass