from jaxtyping import PyTree

import jax
import equinox as eqx
import parax as prx
from equinox import field as field
from parax import unwrap as unwrap, as_free as as_free, as_frozen as as_frozen

def partition(model: PyTree):
    """
    Splits a PyTree into 4 separate trees: variables, constants, 
    bare inexact arrays, and static Python structure.
    """
    return eqx.partition(model, eqx.is_inexact_array, is_leaf=prx.is_constant)