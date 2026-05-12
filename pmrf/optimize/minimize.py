from typing import Callable

import jax.numpy as jnp
import equinox as eqx
import eqxpress as ex
import parax as prx

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.problem import Problem
from pmrf.optimize.base import AbstractMinimizer, minimize as base_minimize
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.backends.scipy import ScipyMinimize


def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: AbstractMinimizer = ScipyMinimize(),
    max_iter: int | None = 1024,
    search_space: str = 'base',
    **kwargs,
) -> OptimizeResult:
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
        If the parameters contain bounds, they are used in a bounded optimization unless `use_bounds` is False.
        If the parameters do not contain bounds, their limits are set to infinity.
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
    if search_space != 'base':
        raise Exception('Only base search space is currently supported')
    
    if isinstance(objective, list):
        objective = ex.Sum([c if isinstance(c, eqx.Module) else prx.Static(ex.Lambda(c)) for c in objective])
    else:
        objective = objective if isinstance(objective, eqx.Module) else prx.Static(ex.Lambda(objective))
    
    # Create the combined problem
    problem = Problem(model=model, frequency=frequency, evaluator=objective)
    
    # Run the optimization
    opt_problem, payload, metrics = base_minimize(
        lambda p, _args: p(),
        y0=problem,
        solver=solver,
        max_iter=max_iter,
        **kwargs,
    )
    
    # Return the results
    results = OptimizeResult(
        model=opt_problem.model,
        objective=opt_problem.evaluator,
        objective_value=opt_problem(),
        metrics=metrics,
    )
    return results