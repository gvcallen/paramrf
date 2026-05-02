from typing import Callable

import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import eqxpress as ex
import parax.optimize as prxo

from pmrf.core import Model, Frequency, Problem
from pmrf.optimize.base import OptimizeResult
from pmrf.optimize.scipy import ScipyMinimize


def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: optx.AbstractMinimiser | prxo.AbstractMinimizer = ScipyMinimize(),
    max_steps: int | None = 1024,
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
    solver : optx.AbstractMinimiser | AbstractBackendMinimizer, default=ScipyMinimize()
        The optimizer to use. Can be either an instance of :class:`pmrf.optimize.ScipyMinimize`,
        a minimizer from Optimistix, or a jaxopt solver class (or a functools.partial of a jaxopt solver class).
    max_steps : int
        The maximum number of iterations to take. 
    **kwargs
        Additional arguments to forward to `parax.optimize.minimize`.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    if isinstance(objective, list):
        objective = ex.Sum([c if isinstance(c, eqx.Module) else ex.Lambda(c) for c in objective])
    else:
        objective = objective if isinstance(objective, eqx.Module) else ex.Lambda(objective)
    
    # Create and validate the problem
    problem = Problem(model=model, frequency=frequency, evaluator=objective)
    
    # Run the optimization
    parax_results = prxo.minimize(
        lambda p, _args: p(),
        solver=solver,
        y0=problem,
        max_steps=max_steps,
        **kwargs,
    )
    opt_problem: Problem = parax_results.model
    
    # Return the results
    results = OptimizeResult(
        model=opt_problem.model,
        objective=opt_problem.evaluator,
        objective_value=parax_results.final_value,
        solver_results=parax_results,
    )
    return results