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
        The RF model holding the final optimized parameters.
    objective_fn : eqx.Module
        The objective function (e.g., :class:`pmrf.evaluators.TargetLoss`)
        used to calculate the objective during optimization. If the objective was an evaluator
        with hyper-parameters, then this contains the optimized objective model.
    value : jnp.ndarray
        The final objective function value achieved by the optimizer.
    solver_results : Any
        The underlying solution object returned by the solver, if any.
    """
    model: Model
    objective_fn: Callable
    value: jnp.ndarray
    solver_results: Any = None