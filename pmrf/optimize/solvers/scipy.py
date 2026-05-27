"""
SciPy optimization wrappers.
"""

from copy import copy
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

DEBUG_PRINT = False

# JaxOpt SciPy wrapper
class ScipyMinimize(AbstractBoundedMinimizer):
    """
    A wrapper around SciPy's :func:`scipy.optimize.minimize`.
    """
    method: str | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    show_progress: bool = eqx.field(static=True, default=True)
    use_grad: bool | None = eqx.field(static=True, default=None)

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

        # Auto-detect gradient requirement
        if use_grad is None:
            gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
            if method is not None and (method.lower() in gradient_free_methods):
                use_grad = False
            else:
                use_grad = True

        # Route to JAXopt if using Autodiff
        if use_grad:
            return self._run_jaxopt(fn, y0, args=args, bounds=bounds, max_iter=max_iter, **kwargs)

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
            
            if DEBUG_PRINT:
                print(f"scipy_bounds = {scipy_bounds}")

        def flat_fn(_flat_y):
            return fn(unravel_fn(_flat_y), args)
            
        val_only_fn = jax.jit(flat_fn)
        current_loss = [np.inf]

        def objective_no_grad(x_np):
            loss = val_only_fn(jnp.array(x_np))
            loss_float = float(loss)
            current_loss[0] = loss_float
            
            if DEBUG_PRINT:
                print(f"params = {x_np}, loss = {loss_float}")
                
            return loss_float

        pbar = None
        if self.show_progress:
            maxiter = scipy_options.get("maxiter", None)
            pbar = tqdm(total=maxiter, desc=f"SciPy {method} (No Grad)")

        def callback(*cb_args, **cb_kwargs):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{current_loss[0]:.3g}")

        try:
            res = scipy_minimize(
                objective_no_grad, 
                np.array(flat_y, dtype=np.float64), 
                jac=False,
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

        return MinimizeResult(y=unravel_fn(jnp.array(res.x)), success=bool(res.success), metrics=res)
    

    def _run_jaxopt(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
        from jaxopt import ScipyBoundedMinimize as JaxOptScipyBoundedMinimize

        pbar = None
        if self.show_progress:
            desc = f"SciPy {self.method}" if self.method is not None else "SciPy (default)"
            pbar = tqdm(total=max_iter, desc=desc)
        def update_pbar(loss_val):
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss_val:.4e}"})

        def wrapped_fn(params, extra_args):
            loss = fn(params, extra_args)
            if self.show_progress:
                jax.debug.callback(update_pbar, loss)
            return loss

        solver = JaxOptScipyBoundedMinimize(
            method=self.method,
            tol=self.tol,
            options=copy(self.options), 
            maxiter=max_iter,
            fun=wrapped_fn,
        )
        
        try:
            y_opt, state = solver.run(y0, bounds, args, **kwargs)
        finally:
            if pbar is not None:
                pbar.close()

        return MinimizeResult(y=y_opt, success=bool(state.success), metrics=state)


# Custom SciPy wrapper
class ScipyMinimize(AbstractBoundedMinimizer):
    """
    A wrapper around SciPy's :func:`scipy.optimize.minimize`.
    """
    method: str | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    show_progress: bool = eqx.field(static=True, default=True)
    use_grad: bool | None = eqx.field(static=True, default=None)

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

        if use_grad is None:
            gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
            if method is not None and (method.lower() in gradient_free_methods):
                use_grad = False
            else:
                use_grad = True

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
            
            if DEBUG_PRINT:
                print(f"scipy_bounds = {scipy_bounds}")

        # Capture `args` via closure to prevent JIT tracing errors
        def flat_fn(_flat_y):
            return fn(unravel_fn(_flat_y), args)
            
        # JIT compile BOTH paths
        val_and_grad_fn = jax.jit(jax.value_and_grad(flat_fn))
        val_only_fn = jax.jit(flat_fn)

        # State containers to pass data to callbacks and prevent log spam
        current_loss = [np.inf]
        nan_logged = [False] 

        def objective_with_grad(x_np):
            loss, grad = val_and_grad_fn(jnp.array(x_np))
            
            # FIX 3: Cast loss to native python float, grad to flat np.float64
            loss_float = float(loss)
            grad_np = np.asarray(grad, dtype=np.float64)
            current_loss[0] = loss_float
            
            if DEBUG_PRINT:
                print(f"params = {x_np}")
                print(f"loss = {loss_float}, grad_np = {grad_np}")
                
            if not nan_logged[0] and np.isnan(loss_float):
                logging.warning(f"Loss value was NaN with parameters = {x_np}")
                
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

        # Setup the progress bar and callback
        pbar = None
        if self.show_progress:
            maxiter = scipy_options.get("maxiter", None)
            pbar = tqdm(total=maxiter, desc=f"SciPy {method}")

        def callback(*cb_args, **cb_kwargs):
            if pbar is not None:
                pbar.update(1)
                # Ensure we format the native float, not a JAX array
                pbar.set_postfix(loss=f"{current_loss[0]:.3g}")

        # Run the optimizer
        try:
            res = scipy_minimize(
                obj_func, 
                np.array(flat_y, dtype=np.float64), 
                jac=use_grad, 
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

        return MinimizeResult(y=unravel_fn(jnp.array(res.x)), success=bool(res.success), metrics=res)
    

    def _run_jaxopt(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
        from jaxopt import ScipyBoundedMinimize as JaxOptScipyBoundedMinimize

        pbar = None
        if self.show_progress:
            desc = f"SciPy {self.method}" if self.method is not None else "SciPy (default)"
            pbar = tqdm(total=max_iter, desc=desc)
        def update_pbar(loss_val):
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss_val:.4e}"})

        def wrapped_fn(params, extra_args):
            loss = fn(params, extra_args)
            if self.show_progress:
                jax.debug.callback(update_pbar, loss)
            return loss

        solver = JaxOptScipyBoundedMinimize(
            method=self.method,
            tol=self.tol,
            options=copy(self.options), 
            maxiter=max_iter,
            fun=wrapped_fn,
        )
        
        try:
            y_opt, state = solver.run(y0, bounds, args, **kwargs)
        finally:
            if pbar is not None:
                pbar.close()

        return MinimizeResult(y=y_opt, success=bool(state.success), metrics=state)