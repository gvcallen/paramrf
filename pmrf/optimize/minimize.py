import equinox as eqx
import optimistix as optx

from pmrf.core import Model, Frequency, Evaluator, Problem, partition
from pmrf.optimize.result import OptimizeResult
from pmrf.transforms import SigmoidHypercubeTransform

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
    params, static = partition(problem)
    transform = SigmoidHypercubeTransform()

    def cost_fn(param, args):
        problem = eqx.combine(param, static)
        return problem()

    optx_results = optx.minimise(cost_fn, solver, params, **kwargs)
    solved_problem = eqx.combine(optx_results.value, static)

    results = OptimizeResult(
        model=solved_problem.model,
        evaluator=solved_problem.evaluator,
        value=solved_problem(),
        history=optx_results.stats,
        success=(optx_results.result == optx.RESULTS.successful),
    )

    return results
    