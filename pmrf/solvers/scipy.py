import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import optimistix as optx
import equinox as eqx

from pypolychord.settings import PolyChordSettings

class ScipyMinimizer(optx.AbstractMinimiser):
    # Equinox needs strings and non-JAX types to be marked as static
    method: str = eqx.field(static=True)
    use_grad: bool = eqx.field(static=True)

    def __init__(self, method="BFGS", use_grad=True):
        self.method = method
        self.use_grad = use_grad

    def init(self, problem, y, args, options):
        # Our state is a simple boolean flag: has the optimization run yet?
        # False means we are ready to step.
        return jnp.array(False)

    def step(self, problem, y, args, options, state):
        
        # This function runs in standard Python (host side)
        def host_minimize(problem_np, y_np, args_np):
            
            # Helper to convert numpy arrays back to JAX arrays for the objective
            def to_jax(x):
                return jnp.array(x) if isinstance(x, (np.ndarray, np.generic)) else x
            
            # problem and args might be PyTrees of tracers, we rebuild them
            problem_jax = jax.tree_util.tree_map(to_jax, problem_np)
            args_jax = jax.tree_util.tree_map(to_jax, args_np)
            
            def objective(x_np):
                x_jax = jnp.array(x_np)
                val = problem_jax.fn(x_jax, args_jax)
                return np.asarray(val, dtype=np.float64)
            
            def jacobian(x_np):
                x_jax = jnp.array(x_np)
                grad = jax.grad(problem_jax.fn)(x_jax, args_jax)
                return np.asarray(grad, dtype=np.float64)

            jac = jacobian if self.use_grad else None

            # Execute the full Scipy optimization
            res = minimize(
                objective, 
                y_np, 
                jac=jac, 
                method=self.method,
                options={"maxiter": options.get("maxiter", 1000)} if options else {}
            )
            
            # Cast back to the original JAX dtype expected
            return res.x.astype(y_np.dtype)

        # Tell JAX what shape and dtype to expect back from Python
        result_shape_dtype = jax.ShapeDtypeStruct(y.shape, y.dtype)
        
        # pure_callback safely handles passing PyTrees of JAX arrays to NumPy
        y_next = jax.pure_callback(
            host_minimize,
            result_shape_dtype,
            problem,
            y,
            args
        )
        
        # Update state to True, indicating the step is complete
        return y_next, jnp.array(True)

    def terminate(self, problem, y, args, options, state):
        # If state is True, we tell Optimistix to stop looping.
        return state, optx.RESULTS.successful

    def buffers(self, state):
        # We don't need to maintain any internal buffers for Scipy
        return ()