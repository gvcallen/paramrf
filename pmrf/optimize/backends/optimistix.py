import optimistix as optx

from pmrf.optimize.base import AbstractOptimizer
from pmrf.optimize.problem import OptimizeProblem
from pmrf.optimize.result import OptimizeResult
from pmrf.transforms import SigmoidHypercubeTransform

class OptimistixOptimizer(AbstractOptimizer):
    solver: optx.AbstractMinimiser
    options: dict

    def __init__(self, solver: optx.AbstractMinimiser, **options):
        self.solver = solver
        self.options = options

    def solve(self, problem: OptimizeProblem, **kwargs) -> OptimizeResult:
        run_options = {**self.options, **kwargs}

        theta0 = problem.get_initial_guess()
        theta_cost_fn = problem.make_flat_cost_fn()
        transform = SigmoidHypercubeTransform(problem.model.distribution())
        x0 = transform(theta0)
        cost_fn = lambda x, args: theta_cost_fn(transform.inv(x))

        result = optx.minimise(
            cost_fn, 
            self.solver,
            y0=x0, 
            **run_options
        )

        x_opt = result.value 
        theta_opt = transform.inv(x_opt)
        optimized_model = problem.reconstruct(theta_opt)
        final_cost = float(cost_fn(x_opt, None))
        success = (result.result == optx.RESULTS.successful)

        return OptimizeResult(
            model=optimized_model,
            cost=final_cost,
            history={"optimistix_stats": result.stats},
            success=success
        )