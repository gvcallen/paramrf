import parax as prx
from typing import Any

import jax.numpy as jnp

from pmrf.core import Model, Evaluator

class OptimizeResult(prx.Module):
    """
    Standardized return object for parameter routines.

    Attributes
    ----------
    model : Model
        The circuit model holding the finalized, optimized parameter state.
    cost : Evaluator
        The evaluator (e.g., metric, sum of goals) used to calculate the objective.
    value : jnp.ndarray
        The final cost value achieved by the optimizer.
    history : Any
        The underlying solution object returned by the solver.
    """
    model: Model
    cost: Evaluator
    value: jnp.ndarray
    history: Any = None