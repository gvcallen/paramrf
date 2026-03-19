import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import optimistix as optx
import equinox as eqx

class ScipyMinimizer(eqx.Module):
    method: str = eqx.field(static=True, default="L-BFGS-B")
    use_grad: bool = eqx.field(static=True, default=True)
    options: dict = eqx.field(static=True, default_factory=dict)

    def __call__(self, fn, y, args, options):
        # 1. Flatten the PyTree 'y' into a 1D JAX array
        flat_y, unravel_fn = ravel_pytree(y)
        
        merged_options = dict(self.options)
        merged_options.update(options)
        bounds = merged_options.pop("bounds", None)

        # 2. Define the internal JAX objective that works on flat arrays
        # We unravel the flat array back into the PyTree 'fn' expects
        @jax.jit
        def flat_fn(_flat_y, _args):
            return fn(unravel_fn(_flat_y), _args)

        val_and_grad_fn = jax.value_and_grad(flat_fn)

        def objective_with_grad(x_np):
            # SciPy provides NumPy, we provide JAX
            loss, grad = val_and_grad_fn(jnp.array(x_np), args)
            return np.asarray(loss, dtype=np.float64), np.asarray(grad, dtype=np.float64)

        def objective_no_grad(x_np):
            loss = flat_fn(jnp.array(x_np), args)
            return np.asarray(loss, dtype=np.float64)

        obj_func = objective_with_grad if self.use_grad else objective_no_grad

        # 3. Optimize on the host
        res = scipy_minimize(
            obj_func, 
            np.array(flat_y), 
            jac=self.use_grad, 
            method=self.method,
            bounds=bounds,
            options=merged_options,
        )

        # 4. Map results to Optimistix Solution struct
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
            state=None
        )