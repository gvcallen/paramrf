import logging

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, Bounds
import equinox as eqx

from optimize.base import AbstractOptimizer
from pmrf.optimize.problem import OptimizeProblem
from pmrf.optimize.result import OptimizeResult

class ScipyOptimizer(AbstractOptimizer):
    method: str
    use_jac: bool
    use_hess: bool
    options: dict

    def __init__(self, method: str = "L-BFGS-B", use_jac: bool = True, use_hess: bool = False, **options):
        self.method = method
        self.use_jac = use_jac
        self.use_hess = use_hess
        self.options = options

    def solve(self, problem: OptimizeProblem, logger: logging.Logger | None = None, **kwargs) -> OptimizeResult:
        logger = logger or logging.getLogger(__name__)
        logger.info(f"Starting SciPy minimize optimization ({self.method}) in [0, 1] space...")

        run_options = {**self.options, **kwargs}
        
        # 1. Get the probability-space cost function and initial guess
        prob_cost = problem.make_prob_cost_fn()
        initial_guess = np.array(problem.flat_prob_initial_guess)

        # 2. Setup strict 0 to 1 bounds for SciPy
        lower_bounds = np.zeros_like(initial_guess)
        upper_bounds = np.ones_like(initial_guess)

        if self.use_jac:
            jax_val_and_grad = eqx.filter_jit(jax.value_and_grad(prob_cost))
            
            def scipy_objective(x_np: np.ndarray):
                val, grad = jax_val_and_grad(jnp.array(x_np))
                return float(val), np.array(grad, dtype=np.float64)
            run_options['jac'] = True
        else:
            jax_val = eqx.filter_jit(prob_cost)

            def scipy_objective(x_np: np.ndarray):
                val = jax_val(jnp.array(x_np))
                return float(val)
            run_options['jac'] = False

        if self.use_hess:
            jax_hessian = eqx.filter_jit(jax.hessian(prob_cost))
            
            def scipy_hessian_fn(x_np: np.ndarray, *args):
                hess = jax_hessian(jnp.array(x_np))
                return np.array(hess, dtype=np.float64)
            run_options['hess'] = scipy_hessian_fn

        # 3. Execute SciPy in the perfectly scaled probability space
        scipy_result = minimize(
            scipy_objective, 
            initial_guess, 
            bounds=Bounds(lower_bounds, upper_bounds), 
            method=self.method,
            **run_options
        )

        logger.info(
            f"Optimization finished: {scipy_result.message} "
            f"(Cost: {scipy_result.fun:.2e}, Iterations: {scipy_result.nfev})"
        )

        # 4. Map the winning [0, 1] array back to Physical Reality, then reconstruct PyTree
        best_u = jnp.clip(jnp.array(scipy_result.x), 1e-7, 1.0 - 1e-7)
        best_physical_x = problem.model.distribution().icdf(best_u)
        optimized_model = problem.reconstruct_fn(best_physical_x)
        
        return OptimizeResult(
            model=optimized_model,
            cost=float(scipy_result.fun),
            history={"scipy_result": scipy_result},
            success=scipy_result.success
        )