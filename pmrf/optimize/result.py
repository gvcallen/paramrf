from typing import Any, Callable

import jax.numpy as jnp
import parax as prx

from pmrf.core import Model, Frequency

class OptimizeResult(prx.Module):
    """
    The result of an optimization run.

    Attributes
    ----------
    model : Model
        The RF model holding the final optimized parameters.
    objective_fn : Callable[[Model, Frequency], jnp.ndarray]
        The objective function (e.g., :class:`pmrf.evaluators.TargetLoss`)
        used to calculate the objective during optimization. If the objective was an module
        with hyper-parameters, then this contains the optimized objective model.
    value : jnp.ndarray
        The final objective function value achieved by the optimizer.
    solver_results : Any
        The underlying solution object returned by the solver, if any.
    """
    model: Model
    objective_fn: Callable[[Model, Frequency], jnp.ndarray]
    value: jnp.ndarray
    solver_results: Any = None