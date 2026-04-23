"""
Base optimization functions and classes.
"""
from typing import Any, Callable

import jax.numpy as jnp
import optimistix as optx
import parax as prx

from pmrf.core import Model, Frequency

class OptimizeResult(prx.Module):
    """
    The result of an optimization run.
    """
    #: The RF model holding the final optimized parameters.
    model: Model

    #: The objective function (e.g., :class:`pmrf.evaluators.TargetLoss`)
    #: used to calculate the objective during optimization. If the objective was a module
    #: with hyper-parameters, then this contains the optimized objective model.
    objective: Callable[[Model, Frequency], jnp.ndarray]

    #: The final objective function value achieved by the optimizer.
    objective_value: jnp.ndarray
    
    #: The underlying results object returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    solver_results: Any = None

    
def is_optimizer(x):
    """
    Returns if a solver is suitable for frequentist optimization in :mod:`pmrf.optimize`.

    Returns `True` for :class:`pmrf.optimize.ScipyMinimize` and :class:`optimistix.AbstractMinimiser`.
    """
    from pmrf.optimize.minimize import ScipyMinimize
    return isinstance(x, ScipyMinimize | optx.AbstractMinimiser)