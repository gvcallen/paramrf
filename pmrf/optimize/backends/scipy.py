import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, Bounds
import equinox as eqx

from optimize.base import AbstractMinimizer
from pmrf.optimize.problem import OptimizeProblem
from pmrf.optimize.result import OptimizeResult
from pmrf.transforms import SigmoidHypercubeTransform

class ScipyOptimizer(AbstractMinimizer):
    method: str
    use_jac: bool
    use_hess: bool
    options: dict

    def __init__(self, method: str = "L-BFGS-B", use_jac: bool = True, use_hess: bool = False, **options):
        self.method = method
        self.use_jac = use_jac
        self.use_hess = use_hess
        self.options = options

    def solve(self, problem: OptimizeProblem, **kwargs) -> OptimizeResult:
        run_options = {**self.options, **kwargs}
        
        theta0 = problem.get_initial_guess()
        theta_cost_fn = problem.make_flat_cost_fn()
        transform = SigmoidHypercubeTransform(problem.model.distribution())
        x0 = transform(theta0)
        cost_fn = lambda x, args: theta_cost_fn(transform.inv(x))

        if self.use_jac:
            jax_val_and_grad = eqx.filter_jit(jax.value_and_grad(cost_fn))
            
            def scipy_objective(x_np: np.ndarray):
                val, grad = jax_val_and_grad(jnp.array(x_np))
                return float(val), np.array(grad, dtype=np.float64)
            run_options['jac'] = True
        else:
            jax_val = eqx.filter_jit(cost_fn)

            def scipy_objective(x_np: np.ndarray):
                val = jax_val(jnp.array(x_np))
                return float(val)
            run_options['jac'] = False

        if self.use_hess:
            jax_hessian = eqx.filter_jit(jax.hessian(cost_fn))
            
            def scipy_hessian_fn(x_np: np.ndarray, *args):
                hess = jax_hessian(jnp.array(x_np))
                return np.array(hess, dtype=np.float64)
            run_options['hess'] = scipy_hessian_fn

        scipy_result = minimize(
            scipy_objective, 
            np.array(x0), 
            bounds=Bounds(np.zeros_like(x0), np.ones_like(x0)),
            method=self.method,
            **run_options
        )

        xopt = jnp.array(scipy_result.x)
        optimized_model = problem.reconstruct(transform.inv(xopt))
        
        return OptimizeResult(
            model=optimized_model,
            cost=float(scipy_result.fun),
            history={"scipy_result": scipy_result},
            success=scipy_result.success
        )