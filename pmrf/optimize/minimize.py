from typing import Callable

import jax
import numpy as np

import equinox as eqx
import optimistix as optx
import parax as prx
from parax.transforms import ParameterTransform, IdentityTransform, LowerPercentile, UpperPercentile

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.evaluators import Summed

def minimize(
    model: Model,
    frequency: Frequency,
    cost: Evaluator | list[Evaluator],
    solver: optx.AbstractMinimiser | Callable = ScipyMinimizer(),
    *,
    transform: ParameterTransform = IdentityTransform(),
    **kwargs,
) -> OptimizeResult:
    if isinstance(cost, list):
        cost = Summed(cost)
    
    problem = Problem(model, frequency, cost)
    return minimize_problem(problem, solver, transform=transform, **kwargs)
        
def minimize_problem(
    problem: Problem,
    solver: optx.AbstractMinimiser | Callable = ScipyMinimizer(),
    *,
    transform: ParameterTransform = IdentityTransform(),
    **kwargs,
) -> OptimizeResult:
    # Apply the transform
    params, static = prx.partition(problem)
    transformed_params, transformed_static = jax.tree.map(transform, (params, static), is_leaf=prx.is_free_param)
    
    def obj_fn(params, _args):
        full_transformed = eqx.combine(params, transformed_static)
        # Invert the mapping across ONLY the problem tree to evaluate physics
        full_physical = jax.tree.map(
            transform.inv, full_transformed, is_leaf=prx.is_free_param
        )
        return full_physical()
    
    # 3. Routing
    if isinstance(solver, optx.AbstractMinimiser):
        solution = optx.minimise(obj_fn, solver, transformed_params, **kwargs)
    else:
        # 1. Extract or Unpack Bounds as PyTrees
        if 'bounds' in kwargs:
            lower_tree, upper_tree = kwargs.pop('bounds')
        else:
            # Automatically build the bound PyTrees from the priors
            lower_tree = jax.tree.map(LowerPercentile(0.999), problem, is_leaf=prx.is_free_param)
            upper_tree = jax.tree.map(UpperPercentile(0.999), problem, is_leaf=prx.is_free_param)
            (lower_tree, upper_tree), _ = prx.partition((lower_tree, upper_tree))
            transformed_lower, transformed_upper = jax.tree.map(transform, (lower_tree, upper_tree), is_leaf=prx.is_free_param)
            kwargs['bounds'] = (transformed_lower, transformed_upper)
            
        if kwargs.get('has_aux', False):
            raise Exception("Auxiliary data not supported for host solvers")
            
        solution = solver(obj_fn, transformed_params, args=None, options=kwargs)

    # 4. Get the solved problem
    solved_transformed_problem = eqx.combine(solution.value, transformed_static)
    solved_problem = jax.tree.map(
        transform.inv, solved_transformed_problem, is_leaf=prx.is_free_param
    )

    # 5. Standardize the results
    results = OptimizeResult(
        model=solved_problem.model,
        cost=solved_problem.evaluator,
        value=solved_problem(),
        history=solution.stats,
        success=(solution.result == optx.RESULTS.successful),
    )
    return results