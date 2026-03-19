import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import optimistix as optx
import equinox as eqx

class ScipyMinimizer(eqx.Module):
    method: str = eqx.field(static=True, default="L-BFGS-B")
    use_grad: bool = eqx.field(static=True, default=True)
    options: dict = eqx.field(static=True, default_factory=dict)

    def __call__(self, fn, y, args, options):
        merged_options = dict(self.options)
        merged_options.update(options)
        
        # Pop bounds so Scipy doesn't choke on them in the options dict
        bounds = merged_options.pop("bounds", None)

        # 1. JIT compile the JAX functions for blazing fast host-to-device calls
        val_and_grad_fn = jax.jit(jax.value_and_grad(fn))
        just_val_fn = jax.jit(fn)

        def objective_with_grad(x_np):
            x_jax = jnp.array(x_np)
            loss, grad = val_and_grad_fn(x_jax, args)
            return np.asarray(loss, dtype=np.float64), np.asarray(grad, dtype=np.float64)

        def objective_no_grad(x_np):
            x_jax = jnp.array(x_np)
            loss = just_val_fn(x_jax, args)
            return np.asarray(loss, dtype=np.float64)

        obj_func = objective_with_grad if self.use_grad else objective_no_grad

        # 2. Run Scipy entirely on the host
        res = scipy_minimize(
            obj_func, 
            np.array(y), 
            jac=self.use_grad, 
            method=self.method,
            bounds=bounds,
            options=merged_options,
        )

        # 3. Map SciPy's exit status to Optimistix's expected ENUM
        if res.success:
            result_state = optx.RESULTS.successful
        else:
            # Map standard Scipy failures (e.g., iteration limit)
            result_state = optx.RESULTS.max_steps_reached

        # 4. Pack the output into an Optimistix-compatible result object
        return optx.Solution(
            value=jnp.array(res.x, dtype=y.dtype),
            result=result_state,
            stats={
                "num_steps": res.nit, 
                "num_evals": res.nfev, 
                "message": res.message, 
                "loss": res.fun
            },
            state=None
        )