from typing import Callable, Any
import equinox as eqx
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency


class OptimizeResult(eqx.Module):
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
    #: Not saved to file.
    metrics: Any = None