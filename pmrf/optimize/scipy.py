"""
Built-in optimization solvers/wrappers.
"""
import logging

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import optimistix as optx
import equinox as eqx
from tqdm.auto import tqdm

from pmrf.optimize.base import AbstractBackendMinimizer

DEBUG_PRINT = False

class ScipyMinimize(AbstractBackendMinimizer):
    """
    A JAX-wrapped optimizer using :func:`scipy.optimize.minimize`.

    Acts as an adapter layer between PyTrees and SciPy's required flat 1D NumPy arrays.
    Handles automatic differentiation via `jax.value_and_grad`.

    Attributes
    ----------
    method : str
        The SciPy solver method (default: "L-BFGS-B" for bounded problems).
    use_grad : bool
        Whether to calculate exact gradients via JAX to pass to the SciPy Jacobian hook.
    tol : float or None
        Tolerance for termination. When `tol` is specified, the selected minimization 
        algorithm sets some relevant solver-specific tolerance(s) equal to `tol`.
    options : dict
        Standard SciPy minimizer options (e.g., 'maxiter', 'ftol', 'disp').
    show_progress : bool
        Whether to display a tqdm progress bar during optimization.
    """
    method: str = eqx.field(static=True, default="L-BFGS-B")
    use_grad: bool = eqx.field(static=True, default=True)
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    show_progress: bool = eqx.field(static=True, default=True)

    @property
    def supports_bounds(self) -> bool:
        return True

    def __call__(self, fn, y, args, options) -> optx.Solution:
        method = self.method
        options = options or {}
        
        lower_tree = options.pop('lower', None)
        upper_tree = options.pop('upper', None)
        
        if 'bounds' in options:
            raise Exception("Bounds should not be passed under scipy minimize options.")

        gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
        use_grad = self.use_grad and (method.lower() not in gradient_free_methods)

        # 1. Flatten the PyTree 'y' into a 1D JAX array
        flat_y, unravel_fn = ravel_pytree(y)
        
        scipy_options = dict(self.options)
        
        # 2. Extract and flatten the bounds PyTrees natively
        scipy_bounds = None
        
        if lower_tree is not None and upper_tree is not None:
            flat_lower, _ = ravel_pytree(lower_tree)
            flat_upper, _ = ravel_pytree(upper_tree)
            
            scipy_bounds = list(zip(np.array(flat_lower), np.array(flat_upper)))        
            
            if DEBUG_PRINT:
                print(f"scipy_bounds = {scipy_bounds}")

        # 3. Define the internal JAX objective that unravels the flat array dynamically
        def flat_fn(_flat_y, _args):
            return fn(unravel_fn(_flat_y), _args)
            
        val_and_grad_fn = jax.jit(jax.value_and_grad(flat_fn))

        # State containers to pass data to callbacks and prevent log spam
        current_loss = [np.inf]
        nan_logged = [False] 

        def objective_with_grad(x_np):
            loss, grad = val_and_grad_fn(jnp.array(x_np), args)
            loss_np = np.asarray(loss, dtype=np.float64)
            grad_np = np.asarray(grad, dtype=np.float64)
            current_loss[0] = loss_np
            
            if DEBUG_PRINT:
                print(f"params = {x_np}")
                print(f"loss = {loss_np}, grad_np = {grad_np}")
                
            if not nan_logged[0] and np.any(np.isnan(loss_np)):
                logging.warning("Loss value was NaN")
                nan_logged[0] = True
            
            if not nan_logged[0] and np.any(np.isnan(grad_np)):
                logging.warning("Loss gradients were NaN")
                nan_logged[0] = True
            
            return loss_np, grad_np

        def objective_no_grad(x_np):
            loss = flat_fn(jnp.array(x_np), args)
            loss_np = np.asarray(loss, dtype=np.float64)
            current_loss[0] = loss_np
            return loss_np

        obj_func = objective_with_grad if use_grad else objective_no_grad

        # 4. Setup the progress bar and callback
        pbar = None
        if self.show_progress:
            maxiter = scipy_options.get("maxiter", None)
            pbar = tqdm(total=maxiter, desc=f"SciPy {method}")

        def callback(*cb_args, **cb_kwargs):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{current_loss[0]:.3g}")

        # 5. Optimize on the host (CPU)
        try:
            res = scipy_minimize(
                obj_func, 
                np.array(flat_y), 
                jac=use_grad, 
                method=method,
                tol=self.tol,
                bounds=scipy_bounds,  
                options=scipy_options,
                callback=callback,
            )
        finally:
            # Ensure the progress bar closes cleanly even if an error occurs
            if pbar is not None:
                pbar.close()

        # 6. Map the results back to an Optimistix compatible Solution struct
        result_state = optx.RESULTS.successful if res.success else optx.RESULTS.max_steps_reached
        
        return optx.Solution(
            value=unravel_fn(jnp.array(res.x)), 
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