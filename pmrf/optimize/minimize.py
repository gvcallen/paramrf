from typing import Callable

import jax

import equinox as eqx
import optimistix as optx
import parax as prx
from distreqx.bijectors import AbstractBijector
from parax.bijectors import Inverse, Identity

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.evaluators import Sum


def minimize(
    cost: Evaluator | list[Evaluator],
    model: Model,
    frequency: Frequency,
    solver: optx.AbstractMinimiser | Callable = ScipyMinimizer(),
    *,
    transform: AbstractBijector | Callable[[prx.Parameter], AbstractBijector] | None = None,
    max_steps: int = 512,
    **kwargs,
) -> OptimizeResult:
    """
    Minimizes a given cost function for a model over a frequency range.
    
    The cost function can have its own hyper-parameters, and is returned in ``result.cost``.

    Parameters
    ----------
    cost : Evaluator | list[Evaluator]
        The objective function to minimize. If a list of Evaluators (e.g., Goals) 
        is provided, they are automatically summed.
    model : Model
        The RF model containing the parameters to be optimized.
    frequency : Frequency
        The frequency sweep over which the cost should be evaluated.
    solver : optx.AbstractMinimiser | Callable, default=ScipyMinimizer()
        The optimization backend to use. Defaults to the host-based SciPy L-BFGS-B.
    transform : distreqx.bijectors.AbstractBijector, default=None
        An invertible transformation to apply to all model parameters before optimization.
    max_steps : int, default=256
        The maximum number of steps/iterations the underlying solver can take.
    **kwargs : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    if isinstance(cost, list):
        cost = Sum(cost)
    
    problem = Problem(cost, model, frequency)    
    
    # Helper to dynamically resolve the bijector for a given parameter
    def get_bijector(p: prx.Parameter) -> AbstractBijector:
        return transform(p) if callable(transform) else transform
    
    def apply_inverse(orig_x, trans_x):
        if isinstance(orig_x, prx.Parameter):
            inv_bij = Inverse(get_bijector(orig_x))
            return trans_x.transformed(inv_bij)
        return trans_x
    
    # 1. Apply the parameter space transform dynamically
    if transform is not None:
        transformed_problem = jax.tree.map(
            lambda x: x.transformed(get_bijector(x)) if isinstance(x, prx.Parameter) else x,
            problem,
            is_leaf=prx.is_free_param
        )
    else:
        transformed_problem = problem
    transformed_params, transformed_static = prx.partition(transformed_problem)
    
    def obj_fn(transformed_params, _args):
        full_transformed = eqx.combine(transformed_params, transformed_static)
        
        # Map over both the source of truth (problem) and the solver state simultaneously
        if transform is not None:
            full_physical = jax.tree.map(
                apply_inverse,
                problem,
                full_transformed,
                is_leaf=prx.is_free_param
            )
        else:
            full_physical = full_transformed
        return full_physical()
    
    # 2. Routing logic for bounding and solver execution
    if isinstance(solver, ScipyMinimizer):
        if 'bounds' in kwargs:
            lower_tree, upper_tree = kwargs.pop('bounds')
        else:
            def lower_percentile(x):
                return x.with_value(x.distribution.icdf(0.01)) if isinstance(x, prx.Parameter) else x
            def upper_percentile(x):
                return x.with_value(x.distribution.icdf(0.99)) if isinstance(x, prx.Parameter) else x
            
            lower_tree = jax.tree.map(lower_percentile, problem, is_leaf=prx.is_free_param)
            upper_tree = jax.tree.map(upper_percentile, problem, is_leaf=prx.is_free_param)
            
            # Transform bounds BEFORE partitioning so the PyTree structure matches `problem`
            def apply_bound_transform(bound_val, orig_p):
                if isinstance(orig_p, prx.Parameter):
                    return bound_val.transformed(get_bijector(orig_p))
                return bound_val

            if transform is not None:
                transformed_lower_tree = jax.tree.map(apply_bound_transform, lower_tree, problem, is_leaf=prx.is_free_param)
                transformed_upper_tree = jax.tree.map(apply_bound_transform, upper_tree, problem, is_leaf=prx.is_free_param)
            else:
                transformed_lower_tree = lower_tree
                transformed_upper_tree = upper_tree
            
            # Now strip out static attributes 
            (transformed_lower, transformed_upper), _ = prx.partition((transformed_lower_tree, transformed_upper_tree))
            kwargs['bounds'] = (transformed_lower, transformed_upper)
            
        if kwargs.get('has_aux', False):
            raise Exception("Auxiliary data not supported for host solvers")
            
        kwargs['maxiter'] = max_steps
        solution = solver(obj_fn, transformed_params, args=None, options=kwargs)
    else:
        solution = optx.minimise(obj_fn, solver, transformed_params, max_steps=max_steps, **kwargs)

    # 3. Get the solved problem and reconstruct the physical state
    solved_transformed_problem = eqx.combine(solution.value, transformed_static)
    
    if transform is not None:
        solved_problem = jax.tree.map(
            apply_inverse,
            problem,
            solved_transformed_problem,
            is_leaf=prx.is_free_param
        )
    else:
        solved_problem = solved_transformed_problem

    # 4. Standardize the results
    results = OptimizeResult(
        model=solved_problem.model,
        cost=solved_problem.evaluator,
        value=solved_problem(),
        history=solution,
    )
    
    if isinstance(solver, optx.AbstractMinimiser):
        print(f"Final cost = {results.value}")
    
    return results