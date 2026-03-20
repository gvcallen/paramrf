from typing import Callable

import jax

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
    """
    Minimizes a given cost function for a circuit model over a frequency range.

    This is a convenient wrapper around `minimize_problem` that automatically 
    aggregates lists of design goals into a single scalar loss function.

    Parameters
    ----------
    model : Model
        The RF circuit model containing the parameters to be optimized.
    frequency : Frequency
        The frequency sweep over which the cost should be evaluated.
    cost : Evaluator | list[Evaluator]
        The objective function to minimize. If a list of Evaluators (e.g., Goals) 
        is provided, they are automatically summed.
    solver : optx.AbstractMinimiser | Callable, default=ScipyMinimizer()
        The optimization backend to use. Defaults to the host-based SciPy L-BFGS-B.
    transform : ParameterTransform, default=IdentityTransform()
        The parameter space transformation (e.g., HypercubeTransform) to apply 
        before optimizing.
    **kwargs : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
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
    """
    Core optimization routing engine.

    Handles PyTree state isolation, space transformation (e.g., physical vs. hypercube), 
    bound extraction for host solvers, and delegates the flat-array math to the 
    requested solver backend.

    Parameters
    ----------
    problem : Problem
        The combined state containing the model, frequency, and evaluator.
    solver : optx.AbstractMinimiser | Callable, default=ScipyMinimizer()
        The solver instance (e.g., Optimistix or ScipyMinimizer).
    transform : ParameterTransform, default=IdentityTransform()
        The geometric transformation mapped across the parameter PyTree.
    **kwargs : dict
        Additional solver options. Explicit `bounds` (as PyTrees) can be passed here.

    Returns
    -------
    OptimizeResult
        The solved state, including the un-transformed (physical) circuit model.
    """
    # 1. Apply the parameter space transform (e.g., to the unit hypercube)
    params, static = prx.partition(problem)
    transformed_params, transformed_static = jax.tree.map(
        transform, (params, static), is_leaf=prx.is_free_param
    )
    
    def obj_fn(params, _args):
        full_transformed = eqx.combine(params, transformed_static)
        # Invert the mapping across ONLY the problem tree to evaluate physical metrics
        full_physical = jax.tree.map(
            transform.inv, full_transformed, is_leaf=prx.is_free_param
        )
        return full_physical()
    
    # 2. Routing logic for bounding and solver execution
    if isinstance(solver, optx.AbstractMinimiser):
        solution = optx.minimise(obj_fn, solver, transformed_params, **kwargs)
    else:
        # Extract or Unpack Bounds as PyTrees for host solvers like SciPy
        if 'bounds' in kwargs:
            lower_tree, upper_tree = kwargs.pop('bounds')
        else:
            # Automatically build the bound PyTrees from the distribution priors (99.9% support)
            lower_tree = jax.tree.map(LowerPercentile(0.999), problem, is_leaf=prx.is_free_param)
            upper_tree = jax.tree.map(UpperPercentile(0.999), problem, is_leaf=prx.is_free_param)
            
            # Strip out static attributes so the bound trees match the parameter tree exactly
            (lower_tree, upper_tree), _ = prx.partition((lower_tree, upper_tree))
            
            # Transform the bounds so they sit in the correct optimization space (e.g., [0, 1])
            transformed_lower, transformed_upper = jax.tree.map(
                transform, (lower_tree, upper_tree), is_leaf=prx.is_free_param
            )
            kwargs['bounds'] = (transformed_lower, transformed_upper)
            
        if kwargs.get('has_aux', False):
            raise Exception("Auxiliary data not supported for host solvers")
            
        solution = solver(obj_fn, transformed_params, args=None, options=kwargs)

    # 3. Get the solved problem and reconstruct the physical state
    solved_transformed_problem = eqx.combine(solution.value, transformed_static)
    solved_problem = jax.tree.map(
        transform.inv, solved_transformed_problem, is_leaf=prx.is_free_param
    )

    # 4. Standardize the results
    results = OptimizeResult(
        model=solved_problem.model,
        cost=solved_problem.evaluator,
        value=solved_problem(),
        history=solution.stats,
        success=(solution.result == optx.RESULTS.successful),
    )
    return results