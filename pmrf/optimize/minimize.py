from typing import Callable

import jax
import numpy as np

import equinox as eqx
import optimistix as optx
import parax as prx
from parax.transforms import IdentityTransform, HypercubeTransform, HypercubeLogitTransform

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.constants import SolverSpace
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers import ScipyMinimizer

def minimize(
    model: Model,
    frequency: Frequency,
    cost: Evaluator,
    solver: optx.AbstractMinimiser | Callable = None,
    *,
    space: SolverSpace = None,
    **kwargs,
) -> OptimizeResult:
    problem = Problem(model, frequency, cost)
    
    return minimize_problem(problem, solver, space=space, **kwargs)
        
def minimize_problem(
    problem: Problem,
    solver: optx.AbstractMinimiser | None = None,
    space: SolverSpace = None,
    **kwargs,
) -> OptimizeResult:
    if solver is None:
        solver = ScipyMinimizer()
    if space is None:
        space = 'hypercube'
        
    # Setup the solver space
    if space == 'hypercube':
        transform = HypercubeTransform()
    elif space == 'logit':
        transform = HypercubeLogitTransform()
    else:
        transform = IdentityTransform()
        
    # Transform the problem and define the transformed objective function
    transformed_problem = jax.tree.map(transform, problem, is_leaf=prx.is_valid_param)
    transformed_params, transformed_static = prx.partition(transformed_problem)
    def obj_fn(params, _args):
        # Rebuild and Invert
        full_transformed = eqx.combine(params, transformed_static)
        full_physical = jax.tree.map(transform.inv, full_transformed, is_leaf=prx.is_valid_param)
        return full_physical()
    
    # Routing to the backend
    if isinstance(solver, optx.AbstractMinimiser):
        solution = optx.minimise(obj_fn, solver, transformed_params, **kwargs)
    else:
        # Host solver (like our ScipyMinimizer)
        if 'bounds' not in kwargs and space == 'hypercube':
            D = transformed_problem.num_flat_params()
            kwargs['bounds'] = list(zip(np.zeros(D), np.ones(D)))
        solution = solver(obj_fn, transformed_params, args=None, options=kwargs)    

    # Get the solved problem
    solved_transformed_problem = eqx.combine(solution.value, transformed_static)
    solved_problem = jax.tree.map(transform.inv, solved_transformed_problem, is_leaf=prx.is_valid_param)

    # Standardize the results
    results = OptimizeResult(
        model=solved_problem.model,
        cost=solved_problem.evaluator,
        value=solved_problem(),
        history=solution.stats,
        success=(solution.result == optx.RESULTS.successful),
    )
    return results