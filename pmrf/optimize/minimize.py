from typing import Callable

import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import parax as prx
from distreqx.bijectors import AbstractBijector

from pmrf.core import Model, Frequency, Problem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers import ScipyMinimizer

def is_optimizer(x):
    """
    Returns if a solver is suitable for frequentist optimization in :mod:`pmrf.optimize`.

    Returns ``True`` for ``optimistix.AbstractMinimiser`` and :class:``pmrf.optimize.ScipyMinimizer``.
    """
    return isinstance(x, optx.AbstractMinimiser | ScipyMinimizer)

def minimize(
    objective_fn: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: optx.AbstractMinimiser | Callable = ScipyMinimizer(),
    *,
    max_steps: int = 10000,
    **kwargs,
) -> OptimizeResult:
    """
    Minimizes a given objective function for a model over a frequency range.
    
    The objective function can have its own hyper-parameters, and is returned in ``result.objective``.

    Parameters
    ----------
    objective_fn : Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
        The objective function to minimize. Must be a callable or PyTree with signature
        (model, freq) -> jnp.ndarray. If a list of objectives is provided, they are automatically summed.
        See :meth:``pmrf.evaluators.Goal`` for an easy way to define goal-based objectives e.g. for design.
    model : Model
        The RF model containing the parameters to be optimized.
    frequency : Frequency
        The frequency sweep over which the objective should be evaluated.
    solver : optx.AbstractMinimiser | Callable, default=ScipyMinimizer()
        The optimization backend to use. Defaults to the host-based SciPy L-BFGS-B.
    max_steps : int, default=256
        The maximum number of steps/iterations the underlying solver can take.
    **kwargs : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    if isinstance(objective_fn, list):
        objective_fn = prx.op.Sum([c if isinstance(c, eqx.Module) else prx.op.Lambda(c) for c in objective_fn])
    else:
        objective_fn = objective_fn if isinstance(objective_fn, eqx.Module) else prx.op.Lambda(objective_fn)
    
    problem = Problem(model=model, frequency=frequency, evaluator=objective_fn)   
    if problem.num_flat_params == 0:
        raise Exception("Received no free parameters in `pmrf.optimize.minimize`") 
    
    params, static = prx.partition(problem)
    def obj_fn(transformed_params, _args):
        full_physical = eqx.combine(transformed_params, static)
        return full_physical()
    
    if isinstance(solver, ScipyMinimizer):
        if 'bounds' in kwargs:
            lower_tree, upper_tree = kwargs.pop('bounds')
        else:
            def lower_fn(x):
                if isinstance(x, prx.Parameter):
                    if x.bounds is not None:
                        low = x.bounds[..., 0]
                    elif x.distribution is not None and hasattr(x.distribution, 'icdf'):
                        low = x.distribution.icdf(0.01*jnp.ones_like(x.value))
                    else:
                        low = jnp.full_like(x.value, -jnp.inf)
                    return x.with_value(low)
                return x
            def upper_fn(x):
                if isinstance(x, prx.Parameter):
                    if x.bounds is not None:
                        high = x.bounds[..., 1]
                    elif x.distribution is not None and hasattr(x.distribution, 'icdf'):
                        high = x.distribution.icdf(0.99*jnp.ones_like(x.value))
                    else:
                        high = jnp.full_like(x.value, jnp.inf)
                    return x.with_value(high)
                return x
            
            lower_tree = jax.tree.map(lower_fn, problem, is_leaf=prx.is_free_param)
            upper_tree = jax.tree.map(upper_fn, problem, is_leaf=prx.is_free_param)
            
            # Now strip out static attributes 
            (lower, upper), _ = prx.partition((lower_tree, upper_tree))
            kwargs['bounds'] = (lower, upper)
            
        if kwargs.get('has_aux', False):
            raise Exception("Auxiliary data not supported for host solvers")
            
        kwargs['maxiter'] = max_steps
        solver_results = solver(obj_fn, params, args=None, options=kwargs)
    else:
        solver_results = optx.minimise(obj_fn, solver, params, max_steps=max_steps, **kwargs)

    optimized_problem = eqx.combine(solver_results.value, static)
    results = OptimizeResult(
        model=optimized_problem.model,
        objective_fn=optimized_problem.evaluator,
        value=optimized_problem(),
        solver_results=solver_results,
    )
    
    if isinstance(solver, optx.AbstractMinimiser):
        logging.info(f"Final objective value = {results.value}")
    
    return results