"""
SciPy optimization wrappers.
"""

from typing import Callable, Any

from jaxtyping import PyTree
import equinox as eqx

from pmrf.optimize.base import AbstractBoundedMinimizer, MinimizerPayload

class ScipyMinimize(AbstractBoundedMinimizer):
    """
    A JAX-wrapped optimizer using :func:`scipy.optimize.minimize`.

    Acts as an adapter layer between PyTrees and SciPy's required flat 1D NumPy arrays.
    Handles automatic differentiation implicitly via JAXopt.
    """
    method: str = eqx.field(static=True, default="L-BFGS-B")
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    
    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizerPayload, PyTree]:
        from jaxopt import ScipyBoundedMinimize as JaxOptScipyBoundedMinimize
        
        solver = JaxOptScipyBoundedMinimize(
            method=self.method,
            tol=self.tol,
            options=self.options,
            maxiter=max_iter,
            fun=fn,
        )
        
        y_opt, state = solver.run(y0, args, bounds=bounds, **kwargs)

        payload = MinimizerPayload(y=y_opt)
        return payload, state