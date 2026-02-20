from typing import Sequence
from dataclasses import dataclass, replace

import jax.numpy as jnp
import equinox as eqx

@dataclass
class ParameterConstraint:
    """Base class for symbolic mathematical constraints between parameters."""
    source: str
    targets: tuple[str, ...]

    def __init__(self, source: str, targets: str | Sequence[str]):
        self.source = source
        self.targets = (targets,) if isinstance(targets, str) else tuple(targets)

    def apply(self, source_val: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Must be implemented by subclasses.")

    def with_prefix(self, prefix: str) -> 'ParameterConstraint':
        """Returns a copy of the constraint prefixed for parent-namespace lifting."""
        new_source = f"{prefix}{self.source}"
        new_targets = tuple(f"{prefix}{t}" for t in self.targets)
        return replace(self, source=new_source, targets=new_targets)

class Equal(ParameterConstraint):
    """Equality constraint: target = source"""
    def apply(self, source_val: jnp.ndarray) -> jnp.ndarray:
        return source_val

class Linear(ParameterConstraint):
    """Linear constraint: target = scale * source + offset"""
    scale: float
    offset: float

    def __init__(self, source: str, targets: str | Sequence[str], scale: float = 1.0, offset: float = 0.0):
        super().__init__(source, targets)
        self.scale = float(scale)
        self.offset = float(offset)

    def apply(self, source_val: jnp.ndarray) -> jnp.ndarray:
        return source_val * self.scale + self.offset