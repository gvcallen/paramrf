import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import optimistix as optx
import equinox as eqx

class ScipyMinimizer(eqx.Module):
    method: str = eqx.field(static=True, default="L-BFGS-B")
    use_grad: bool = eqx.field(static=True, default=False)
    options: dict = eqx.field(static=True, default_factory=dict)

    def __call__(self, fn, y, args, options) -> optx.Solution:
        # 1. Flatten the PyTree 'y' into a 1D JAX array
        flat_y, unravel_fn = ravel_pytree(y)
        
        merged_options = dict(self.options)
        merged_options.update(options)
        
        # 2. Extract and flatten the bounds PyTrees
        bounds_trees = merged_options.pop("bounds", None)
        scipy_bounds = None
        
        if bounds_trees is not None:
            lower_tree, upper_tree = bounds_trees
            
            # Flatten the bound trees using the exact same structure logic
            flat_lower, _ = ravel_pytree(lower_tree)
            flat_upper, _ = ravel_pytree(upper_tree)
            
            # SciPy expects a sequence of (min, max) tuples for each parameter
            scipy_bounds = list(zip(np.array(flat_lower), np.array(flat_upper)))

        # 3. Define the internal JAX objective that works on flat arrays
        @jax.jit
        def flat_fn(_flat_y, _args):
            cost = fn(unravel_fn(_flat_y), _args)
            return cost

        val_and_grad_fn = jax.value_and_grad(flat_fn)

        def objective_with_grad(x_np):
            # SciPy provides NumPy, we provide JAX
            loss, grad = val_and_grad_fn(jnp.array(x_np), args)
            return np.asarray(loss, dtype=np.float64), np.asarray(grad, dtype=np.float64)

        def objective_no_grad(x_np):
            loss = flat_fn(jnp.array(x_np), args)
            return np.asarray(loss, dtype=np.float64)

        obj_func = objective_with_grad if self.use_grad else objective_no_grad

        # 4. Optimize on the host
        res = scipy_minimize(
            obj_func, 
            np.array(flat_y), 
            jac=self.use_grad, 
            method=self.method,
            bounds=scipy_bounds,  # Pass the newly zipped flat bounds
            options=merged_options,
        )

        # 5. Map results to Optimistix Solution struct
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