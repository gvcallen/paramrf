"""
SciPy optimization wrappers.
"""

import logging
from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree
import equinox as eqx
from tqdm.auto import tqdm
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from pmrf.optimize.base import AbstractBoundedMinimizer, MinimizeResult

DEBUG = False

class ScipyMinimize(AbstractBoundedMinimizer):
    """
    A wrapper around SciPy's :func:`scipy.optimize.minimize`.
    """
    method: str | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    show_progress: bool = eqx.field(static=True, default=True)
    use_grad: bool | None = eqx.field(static=True, default=None)
    # use_hess: bool | None = eqx.field(static=True, default=None)

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
        options = self.options
        method = self.method
        tol = self.tol
        use_grad = self.use_grad
        use_hess = False

        if use_grad is None:
            gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
            if method is not None and (method.lower() in gradient_free_methods):
                use_grad = False
            else:
                use_grad = True

        if use_hess is None:
            hessian_methods = {'newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr'}
            if method is not None and (method.lower() in hessian_methods):
                use_hess = True
            else:
                use_hess = False

        if 'max_iter' in options:
            raise ValueError("Cannot pass `max_iter` in SciPy options")

        lower_tree, upper_tree = bounds if bounds is not None else (None, None)

        flat_y, unravel_fn = ravel_pytree(y0)
        scipy_options = dict(options)
        scipy_options.setdefault('maxiter', max_iter)

        scipy_bounds = None
        if lower_tree is not None and upper_tree is not None:
            flat_lower, _ = ravel_pytree(lower_tree)
            flat_upper, _ = ravel_pytree(upper_tree)
            scipy_bounds = list(zip(np.array(flat_lower), np.array(flat_upper)))        
            
            if DEBUG:
                print(f"scipy_bounds = {scipy_bounds}")

        def flat_fn(_flat_y):
            return fn(unravel_fn(_flat_y), args)
            
        val_and_grad_fn = jax.jit(jax.value_and_grad(flat_fn))
        val_only_fn = jax.jit(flat_fn)

        current_loss = [np.inf]
        nan_logged = [False] 

        def objective_with_grad(x_np):
            loss, grad = val_and_grad_fn(jnp.array(x_np))
            
            loss_float = float(loss)
            grad_np = np.asarray(grad, dtype=np.float64)
            current_loss[0] = loss_float
            
            if DEBUG:
                print(f"params = {x_np}\nloss = {loss_float}, grad_np = {grad_np}")
                
            if not nan_logged[0] and np.isnan(loss_float):
                logging.warning("Loss value was NaN")
                if DEBUG:
                    np.savetxt("nan_values.txt", x_np)
                nan_logged[0] = True
            
            if not nan_logged[0] and np.any(np.isnan(grad_np)):
                logging.warning("Loss gradients were NaN")
                nan_logged[0] = True
            
            return loss_float, grad_np

        def objective_no_grad(x_np):
            loss = val_only_fn(jnp.array(x_np))
            loss_float = float(loss)
            current_loss[0] = loss_float
            return loss_float

        obj_func = objective_with_grad if use_grad else objective_no_grad

        if use_hess:
            hessian_fn = jax.jit(jax.hessian(flat_fn))
            def objective_hess(x_np):
                return np.asarray(hessian_fn(jnp.array(x_np)), dtype=np.float64)
            hess_arg = objective_hess
        else:
            hess_arg = None

        pbar = None
        if self.show_progress:
            maxiter = scipy_options.get("maxiter", None)
            desc = f"SciPy {method}" if method is not None else "SciPy (default)"
            pbar = tqdm(total=maxiter, desc=desc)

        def callback(*cb_args, **cb_kwargs):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{current_loss[0]:.3g}")

        try:
            res = scipy_minimize(
                obj_func, 
                np.array(flat_y, dtype=np.float64), 
                jac=use_grad, 
                hess=hess_arg,
                method=method,
                tol=tol,
                bounds=scipy_bounds,  
                options=scipy_options,
                callback=callback,
                **kwargs,
            )
        finally:
            if pbar is not None:
                pbar.close()

        return MinimizeResult(
            y=unravel_fn(jnp.array(res.x)), 
            success=bool(res.success), 
            metrics=res
        )