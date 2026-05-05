from typing import Callable, Any

import jax.numpy as jnp
import equinox as eqx
import eqxpress as ex
from jaxtyping import PyTree, Scalar
import parax as prx

from pmrf.core import Model, Frequency, Problem
from pmrf.optimize.base import OptimizeResult, AbstractMinimizer, AbstractBoundedMinimizer, MinimizerPayload
from pmrf.optimize.jaxopt import LBFGSB

@eqx.filter_jit
def minimize_parax(
    fn: Callable[[PyTree, Any], Scalar], 
    model: PyTree, 
    solver: AbstractMinimizer,
    args: Any = None,
    max_iter: int = 1024, 
    **kwargs
) -> tuple[PyTree, MinimizerPayload, PyTree]:
    """
    Optimizes a general PyTree potentially containing Parax parameters using either a bounded or unconstrained solver.
    """
    is_bounded = isinstance(solver, AbstractBoundedMinimizer)
    filter_spec = eqx.is_inexact_array
    
    # Extract base values and partition based on solver type
    if is_bounded:
        base_tree = prx.bounded.tree_base(model)
        lower_bounds, upper_bounds = prx.bounded.tree_bounds(model)
        
        params, static = eqx.partition(base_tree, filter_spec, is_leaf=prx.is_constant)
        lower, _ = eqx.partition(lower_bounds, filter_spec, is_leaf=prx.is_constant)
        upper, _ = eqx.partition(upper_bounds, filter_spec, is_leaf=prx.is_constant)
        bounds = (lower, upper)
    else:
        params, static = eqx.partition(model, filter_spec, is_leaf=prx.is_constant)

    # Define the unified objective wrapper for the solver
    def objective(p: PyTree, args: Any) -> Scalar:
        unwrapped_model = prx.unwrap(eqx.combine(p, static))
        return fn(unwrapped_model, args)

    # Run the correct solver execution and reconstruct the final model
    if is_bounded:
        payload, metrics = solver.run(
            fn=objective, y0=params, args=args, bounds=bounds, max_iter=max_iter, **kwargs
        )
        opt_base = eqx.combine(payload.y, static)
        final_model = prx.bounded.tree_update(model, opt_base)
    else:
        payload, metrics = solver.run(
            fn=objective, y0=params, args=args, max_iter=max_iter, **kwargs
        )
        final_model = prx.unwrap(eqx.combine(payload.y, static))

    return final_model, payload, metrics

@eqx.filter_jit
def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: AbstractMinimizer = LBFGSB(),
    max_iter: int | None = 1024,
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
    solver : pmrf.optimize.AbstractMinimizer, default=LBFGSB()
        The optimizer to use. See :type:`pmrf.optimize.AbstractMinimizer`.
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
        objective = ex.Sum([c if isinstance(c, eqx.Module) else ex.Lambda(c) for c in objective])
    else:
        objective = objective if isinstance(objective, eqx.Module) else ex.Lambda(objective)
    
    # Create the combined problem
    problem = Problem(model=model, frequency=frequency, evaluator=objective)
    
    # Run the optimization
    opt_problem, payload, metrics = minimize_parax(
        lambda p, _args: p(),
        solver=solver,
        model=problem,
        max_iter=max_iter,
        **kwargs,
    )
    
    # Return the results
    results = OptimizeResult(
        model=opt_problem.model,
        objective=opt_problem.evaluator,
        objective_value=opt_problem(),
        solver_results=metrics,
    )
    return results