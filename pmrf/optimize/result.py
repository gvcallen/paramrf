from typing import Callable, Any, TypeVar, Generic
import equinox as eqx
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.terms import TermFn


ModelT = TypeVar('ModelT', bound=Model)


class OptimizeResult(eqx.Module, Generic[ModelT]):
    """
    The result of an optimization run.
    """
    #: The RF model holding the final optimized parameters.
    model: ModelT

    #: The terms that were summed to form the objective during optimization. If a term's
    #: evaluator was a module with hyper-parameters, then this contains the optimized evaluator.
    objective: tuple[TermFn, ...]

    #: Whether the optimizer converged.
    success: bool
    
    #: The underlying results object returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    #: Not saved to file.
    metrics: Any = None