from typing import Callable, Sequence, TypeVar

import jax.numpy as jnp

from pmrf.models import Model, validate
from pmrf.frequency import Frequency
from pmrf.problem import Problem
from pmrf.terms import TermLike, as_terms
from pmrf.optimize.base import AbstractMinimizer, run_minimizer
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers.scipy import ScipyMinimize

ModelT = TypeVar('ModelT', bound=Model)

def minimize(
    objective: TermLike | Sequence[TermLike],
    model: ModelT,
    frequency: Frequency | None = None,
    solver: AbstractMinimizer = ScipyMinimize(),
    max_iter: int | None = 1024,
    **kwargs,
) -> OptimizeResult[ModelT]:
    """
    Minimizes a given objective function for a model over a frequency range.
    
    The objective function can have its own hyper-parameters, and is returned in `result.objective`.

    Parameters
    ----------
    objective : TermLike | Sequence[TermLike]
        The objective function to minimize. Can be a function or a callable PyTree
        with optional parameters. If a sequence of objectives is provided,
        they are automatically summed. See :meth:`pmrf.evaluators.Goal`
        for an easy way to define goal-based objectives.

        Each objective may instead be an ``(objective, frequency)`` pair, or a
        :class:`pmrf.Term`, binding it to its own frequency sweep rather than the
        shared one. This allows a single parameter set to be optimized against
        several bands at once.
    model : Model
        The RF model containing the parameters to be optimized.
        If the parameters contain bounds and the optimizer supports bounds, these bounds
        are used in a bounded optimization. Otherwise, the bounds are enforced
        via space transformations (bijectors). If the parameters do not contain bounds,
        their limits are set to infinity.
    frequency : Frequency | None, default=None
        The frequency sweep over which the objective should be evaluated. May be
        omitted only if every objective already carries its own frequency.
    solver : pmrf.optimize.AbstractMinimizer, default=ScipyMinimize()
        The optimizer to use.
        See :mod:`pmrf.optimize` for available solvers.
    max_iter : int
        The maximum number of iterations to take.
    **kwargs
        Additional arguments to forward to `parax.optimize.minimize`.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    # Create the combined problem
    problem = Problem(model=model, terms=as_terms(objective, frequency))

    validate(problem)
    
    # Run the optimization
    opt_problem, result = run_minimizer(
        lambda p, _args: p(),
        model=problem,
        solver=solver,
        max_iter=max_iter,
        **kwargs,
    )
    
    # Return the results
    results = OptimizeResult(
        model=opt_problem.model,
        objective=opt_problem.terms,
        success=result.success,
        metrics=result.metrics,
    )
    return results