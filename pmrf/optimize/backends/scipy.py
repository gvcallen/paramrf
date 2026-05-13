"""
SciPy optimization wrappers.
"""

from copy import copy
from typing import Callable, Any

import jax
from jaxtyping import PyTree
import equinox as eqx
from tqdm.auto import tqdm

from pmrf.optimize.base import AbstractBoundedMinimizer, MinimizeResults

class ScipyMinimize(AbstractBoundedMinimizer):
    """
    A JAX-wrapped optimizer using :func:`scipy.optimize.minimize`.

    Acts as an adapter layer between PyTrees and SciPy's required flat 1D NumPy arrays.
    Handles automatic differentiation implicitly via JAXopt.
    """
    method: str = eqx.field(static=True, default="L-BFGS-B")
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
    ) -> tuple[MinimizeResults, PyTree]:
        from jaxopt import ScipyBoundedMinimize as JaxOptScipyBoundedMinimize
        
        # 1. Initialize tqdm if requested
        pbar = None
        if self.show_progress:
            # Note: total is approximate as max_iter usually refers to 
            # iterations, but we track function evaluations.
            pbar = tqdm(total=max_iter, desc=f"SciPy {self.method}")

        # 2. Define the callback to update the bar
        def update_pbar(loss_val):
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss_val:.4e}"})

        # 3. Wrap the objective function
        def wrapped_fn(params, extra_args):
            loss = fn(params, extra_args)
            if self.show_progress:
                # debug.callback allows side-effects (tqdm) in JAX-jitted code
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
            # 4. Ensure the progress bar is closed even if optimization fails
            if pbar is not None:
                pbar.close()

        payload = MinimizeResults(y=y_opt)
        return payload, state