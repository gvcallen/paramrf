"""
SciPy optimization wrappers.
"""

from copy import copy
from typing import Callable, Any

import jax
from jaxtyping import PyTree
import equinox as eqx
from tqdm.auto import tqdm

from pmrf.optimize.base import AbstractBoundedMinimizer, MinimizeResult

class ScipyMinimize(AbstractBoundedMinimizer):
    """
    A wrapper around SciPy's :func:`scipy.optimize.minimize`.
    
    Acts as an adapter layer between JAX's PyTrees and SciPy's required flat 1D NumPy arrays.
    Handles automatic differentiation implicitly via JAXopt.
    
    Parameters
    ----------
    method : str, optional
        Type of solver, pass None to use SciPy defaults.
    tol : float, optional
        Tolerance for termination.
    options : dict, optional
        A dictionary of solver options.
    show_progress : bool, default=True
        Whether to show a progress bar during optimization.
    """
    method: str | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    show_progress: bool = eqx.field(static=True, default=True)
    
    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizeResult, PyTree]:
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

        payload = MinimizeResult(y=y_opt, success=bool(state.success))
        return payload, state