import jax

import equinox as eqx
import optimistix as optx
import parax as prx
from parax.transforms import UnboundedTransform

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
    transform = UnboundedTransform()
    unbounded_problem = jax.tree.map(transform, problem, is_leaf=prx.is_valid_param)
    unbounded_params, unbounded_static = prx.partition(unbounded_problem)

    def cost_fn(unbounbed_params, _args):
        unbounded_problem = eqx.combine(unbounbed_params, unbounded_static)
        problem = jax.tree.map(transform.inv, unbounded_problem, is_leaf=prx.is_valid_param)
        return problem()

    optx_results = optx.minimise(cost_fn, solver, unbounded_params, **kwargs)
    solved_unbounded_problem = eqx.combine(optx_results.value, unbounded_static)
    solved_problem = jax.tree.map(transform.inv, solved_unbounded_problem, is_leaf=prx.is_valid_param)

    results = OptimizeResult(
        model=solved_problem.model,
        evaluator=solved_problem.evaluator,
        value=solved_problem(),
        history=optx_results.stats,
        success=(optx_results.result == optx.RESULTS.successful),
    )

    return results
    