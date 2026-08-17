from typing import Callable, Any, TypeVar, Generic
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

from pmrf.frequency import Frequency
from pmrf.problems import AbstractProblem, problem_terms
from pmrf.terms import TermFn


PyTreeT = TypeVar('PyTreeT', bound=PyTree)


class OptimizeResult(eqx.Module, Generic[PyTreeT]):
    """
    The result of an optimization run.
    """
    #: The solved problem, holding the final optimized parameters.
    problem: AbstractProblem

    #: Whether the optimizer converged.
    success: bool
    
    #: The underlying results object returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    #: Not saved to file.
    metrics: Any = None

    @property
    def model(self) -> PyTreeT:
        """
        The PyTree holding the final optimized parameters.
        """
        return self.problem.model

    @property
    def objective(self) -> tuple[TermFn, ...]:
        """
        The terms that were summed to form the objective. If a term's evaluator was a
        module with hyper-parameters, then this contains the optimized evaluator.
        """
        return problem_terms(self.problem, 'objective')
