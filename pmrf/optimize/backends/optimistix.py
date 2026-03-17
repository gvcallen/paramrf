import logging

import jax.numpy as jnp
import jax.nn as jnn
import optimistix as optx

from optimize.base import AbstractOptimizer
from pmrf.optimize.problem import OptimizeProblem
from pmrf.optimize.result import OptimizeResult

class OptimistixOptimizer(AbstractOptimizer):
    solver: optx.AbstractMinimiser
    options: dict

    def __init__(self, solver: optx.AbstractMinimiser, **options):
        self.solver = solver
        self.options = options

    def solve(self, problem: OptimizeProblem, logger: logging.Logger | None = None, **kwargs) -> OptimizeResult:
        logger = logger or logging.getLogger(__name__)
        logger.info(f"Starting Optimistix optimization ({self.solver.__class__.__name__}) in unbounded space...")

        run_options = {**self.options, **kwargs}

        # 1. Get the unbounded-space cost function and initial guess
        unbounded_cost = problem.make_unbounded_cost_fn()
        initial_guess = problem.flat_unbounded_initial_guess

        # Optimistix expects fn(y, args) -> scalar
        def optx_cost(y, args):
            return unbounded_cost(y)

        # 2. Run the minimizer on the unbounded 1D flat array
        result = optx.minimise(
            optx_cost, 
            self.solver, 
            y0=initial_guess, 
            **run_options
        )

        # 3. Map the winning [-inf, inf] array back to Prob -> Physical -> PyTree
        best_unbounded_y = result.value
        best_u = jnp.clip(jnn.sigmoid(best_unbounded_y), 1e-7, 1.0 - 1e-7)
        best_physical_x = problem.model.distribution().icdf(best_u)
        
        optimized_model = problem.reconstruct_fn(best_physical_x)
        final_cost = float(optx_cost(result.value, None))
        success = (result.result == optx.RESULTS.successful)

        logger.info(
            f"Optimization finished. "
            f"(Cost: {final_cost:.2e}, Success: {success})"
        )

        return OptimizeResult(
            model=optimized_model,
            cost=final_cost,
            history={"optx_stats": result.stats},
            success=success
        )