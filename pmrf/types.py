"""
Type variables and type aliases.
"""
from __future__ import annotations

from typing import TypeAlias

from jaxtyping import (
    ArrayLike as ArrayLike,
    Inexact, Array
)

import parax as prx

# NB these type-hints must be re-defined in __init__ for proper docs

#: The canonical type hint for a float, or a numpy or JAX array.
ArrayLike: TypeAlias = ArrayLike

#: The canonical type hint for a fixed or variable parameter.
#: Parameters should be created using factories in :mod:`pmrf.parameters`,
#: most of which are re-exported at root (e.g. :func:`pmrf.Unconstrained`, :func:`pmrf.Fixed`, :func:`pmrf.Bounded`).
Param: TypeAlias = prx.AbstractVariable | Inexact[Array, "..."]