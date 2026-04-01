import parax as prx
from typing import Any, Callable

import jax.numpy as jnp
from jaxtyping import Array


from pmrf.core import Model

class OptimizeResult(prx.Module):
    """
    Standardized return object for parameter routines.

    Attributes
    ----------
    model : Model
        The circuit model holding the finalized, optimized parameter state.
    cost : eqx.Module
        The evaluator (e.g., metric, sum of goals) used to calculate the objective.
    value : jnp.ndarray
        The final cost value achieved by the optimizer.
    stats : Any
        The stats from the underlying solver.
    """
    model: Model
    cost: Callable
    value: jnp.ndarray
    stats: Any = None