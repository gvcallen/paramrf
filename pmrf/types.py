"""
Type variables and type aliases.
"""
from __future__ import annotations

from typing import TypeAlias

from jaxtyping import (
    ArrayLike as ArrayLike,
)

#: The canonical type hint for a float, or a numpy or JAX array.
ArrayLike: TypeAlias = ArrayLike