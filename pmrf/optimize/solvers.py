import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import optimistix as optx
import equinox as eqx


class ScipyMinimizer(eqx.Module):
    """
    A host-based minimizer utilizing scipy.optimize.minimize.

    Acts as an adapter layer between highly nested JAX PyTrees and SciPy's required
    flat 1D NumPy arrays. Safely handles automatic differentiation via jax.value_and_grad
    and un-packs PyTree boundaries into the exact sequence formatting SciPy demands.

    Attributes
    ----------
    method : str
        The SciPy solver method (default: "L-BFGS-B" for bounded problems).
    use_grad : bool
        Whether to calculate exact gradients via JAX to pass to the SciPy Jacobian hook.
    options : dict
        Standard SciPy minimizer options (e.g., 'maxiter', 'ftol', 'disp').
    """
    method: str = eqx.field(static=True, default="L-BFGS-B")
    use_grad: bool = eqx.field(static=True, default=False)
    options: dict = eqx.field(static=True, default_factory=dict)

    def __call__(self, fn, y, args, options) -> optx.Solution:
        # 1. Flatten the PyTree 'y' into a 1D JAX array
        flat_y, unravel_fn = ravel_pytree(y)
        
        merged_options = dict(self.options)
        merged_options.update(options)
        
        # 2. Extract and flatten the bounds PyTrees natively
        bounds_trees = merged_options.pop("bounds", None)
        scipy_bounds = None
        
        if bounds_trees is not None:
            lower_tree, upper_tree = bounds_trees
            
            # Flatten the bound trees using the exact same structural logic
            flat_lower, _ = ravel_pytree(lower_tree)
            flat_upper, _ = ravel_pytree(upper_tree)
            
            # SciPy expects a sequence of (min, max) tuples for each parameter
            scipy_bounds = list(zip(np.array(flat_lower), np.array(flat_upper)))

        # 3. Define the internal JAX objective that unravels the flat array dynamically
        @jax.jit
        def flat_fn(_flat_y, _args):
            cost = fn(unravel_fn(_flat_y), _args)
            return cost

        val_and_grad_fn = jax.value_and_grad(flat_fn)

        def objective_with_grad(x_np):
            # SciPy provides NumPy arrays, we must provide JAX arrays
            loss, grad = val_and_grad_fn(jnp.array(x_np), args)
            return np.asarray(loss, dtype=np.float64), np.asarray(grad, dtype=np.float64)

        def objective_no_grad(x_np):
            loss = flat_fn(jnp.array(x_np), args)
            return np.asarray(loss, dtype=np.float64)

        obj_func = objective_with_grad if self.use_grad else objective_no_grad

        # 4. Optimize on the host (CPU)
        res = scipy_minimize(
            obj_func, 
            np.array(flat_y), 
            jac=self.use_grad, 
            method=self.method,
            bounds=scipy_bounds,  
            options=merged_options,
        )

        # 5. Map the results back to an Optimistix compatible Solution struct
        result_state = optx.RESULTS.successful if res.success else optx.RESULTS.max_steps_reached
        
        return optx.Solution(
            value=unravel_fn(jnp.array(res.x)), # Reconstruct the original PyTree structure
            result=result_state,
            stats={
                "num_steps": res.nit, 
                "num_evals": res.nfev, 
                "message": res.message, 
                "loss": float(res.fun)
            },
            aux=args,
            state=None
        )