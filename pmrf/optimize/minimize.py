from typing import Callable, TypeVar

import jax.numpy as jnp
import equinox as eqx
import eqxpress as ex
import parax as prx

from pmrf.models import Model, validate
from pmrf.frequency import Frequency
from pmrf.problem import Problem
from pmrf.optimize.base import AbstractMinimizer, run_minimizer
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers.scipy import ScipyMinimize

ModelT = TypeVar('ModelT', bound=Model)

def minimize(
    objective: Callable[[ModelT, Frequency], jnp.ndarray] | list[Callable],
    model: ModelT,
    frequency: Frequency,
    solver: AbstractMinimizer = ScipyMinimize(),
    max_iter: int | None = 1024,
    **kwargs,
) -> OptimizeResult[ModelT]:
    """
    Minimizes a given objective function for a model over a frequency range.
    
    The objective function can have its own hyper-parameters, and is returned in `result.objective`.

    Parameters
    ----------
    objective : Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
        The objective function to minimize. Can be a function or a callable PyTree
        with optional parameters. If a list of objectives is provided,
        they are automatically summed. See :meth:`pmrf.evaluators.Goal`
        for an easy way to define goal-based objectives.
    model : Model
        The RF model containing the parameters to be optimized.
        If the parameters contain bounds and the optimizer supports bounds, these bounds
        are used in a bounded optimization. Otherwise, the bounds are enforced
        via space transformations (bijectors). If the parameters do not contain bounds,
        their limits are set to infinity.
    frequency : Frequency
        The frequency sweep over which the objective should be evaluated.
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
    if isinstance(objective, list):
        objective = ex.Sum([c if isinstance(c, eqx.Module) else prx.Static(ex.Lambda(c)) for c in objective])
    else:
        objective = objective if isinstance(objective, eqx.Module) else prx.Static(ex.Lambda(objective))
    
    # Create the combined problem
    problem = Problem(model=model, frequency=frequency, evaluator=objective)
    
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
        objective=opt_problem.evaluator,
        success=result.success,
        metrics=result.metrics,
    )
    return results