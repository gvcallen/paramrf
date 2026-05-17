from typing import Callable, Any, TypeVar, Generic
import equinox as eqx
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency


ModelT = TypeVar('ModelT', bound=Model)


class OptimizeResult(eqx.Module, Generic[ModelT]):
    """
    The result of an optimization run.
    """
    #: The RF model holding the final optimized parameters.
    model: ModelT

    #: The objective function (e.g., :class:`pmrf.evaluators.TargetLoss`)
    #: used to calculate the objective during optimization. If the objective was a module
    #: with hyper-parameters, then this contains the optimized objective model.
    objective: Callable[[ModelT, Frequency], jnp.ndarray]

    #: Whether the optimizer converged.
    success: bool
    
    #: The underlying results object returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    #: Not saved to file.
    metrics: Any = None