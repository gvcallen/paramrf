import jax

import equinox as eqx
import optimistix as optx
import parax as prx
from parax.transforms import HypercubeLogitTransform

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.optimize.result import OptimizeResult

def minimize(
    model: Model,
    frequency: Frequency,
    cost: Evaluator,
    solver: optx.AbstractMinimiser | None = None,
    **kwargs,
) -> OptimizeResult:
    if solver is None:
        solver = optx.LBFGS()
    
    problem = Problem(model, frequency, cost)
    transform = HypercubeLogitTransform()
    transformed_problem = jax.tree.map(transform, problem, is_leaf=prx.is_valid_param)
    transformed_params, transformed_static = prx.partition(transformed_problem)

    def cost_fn(transformed_params, _args):
        transformed_problem = eqx.combine(transformed_params, transformed_static)
        problem = jax.tree.map(transform.inv, transformed_problem, is_leaf=prx.is_valid_param)
        return problem()

    optx_results = optx.minimise(cost_fn, solver, transformed_params, **kwargs)
    solved_transformed_problem = eqx.combine(optx_results.value, transformed_static)
    solved_problem = jax.tree.map(transform.inv, solved_transformed_problem, is_leaf=prx.is_valid_param)

    results = OptimizeResult(
        model=solved_problem.model,
        evaluator=solved_problem.evaluator,
        value=solved_problem(),
        history=optx_results.stats,
        success=(optx_results.result == optx.RESULTS.successful),
    )

    return results
    