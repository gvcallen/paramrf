"""
SciPy optimization wrappers.
"""

from copy import copy
from typing import Callable, Any

from jaxtyping import PyTree
import equinox as eqx

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

        solver = JaxOptScipyBoundedMinimize(
            method=self.method,
            tol=self.tol,
            options=copy(self.options), # NB must copy otherwise JAXopt modifies our options
            maxiter=max_iter,
            fun=fn,
        )
        
        y_opt, state = solver.run(y0, bounds, args, **kwargs)

        payload = MinimizeResults(y=y_opt)
        return payload, state