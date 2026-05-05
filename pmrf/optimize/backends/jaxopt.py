"""
JAXopt optimization wrappers.
"""

from typing import Callable, Any

import equinox as eqx
from jaxtyping import PyTree
import jaxopt

from pmrf.optimize.base import AbstractBoundedMinimizer, MinimizerPayload

class LBFGSB(AbstractBoundedMinimizer):
    """
    A pure-JAX L-BFGS-B bounded optimizer wrapping :class:`jaxopt.LBFGSB`.
    
    Unlike `ScipyMinimize`, this implementation is written entirely in JAX 
    and can be fully JIT-compiled (e.g., via `eqx.filter_jit`).
    """
    tol: float = eqx.field(static=True, default=1e-3)
    stepsize: float = eqx.field(static=True, default=1.0)
    linesearch: str = eqx.field(static=True, default="zoom")
    
    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None, 
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizerPayload, PyTree]:
        solver = jaxopt.LBFGSB(
            fun=fn,
            tol=self.tol,
            stepsize=self.stepsize,
            maxiter=max_iter,
            linesearch=self.linesearch,
        )
        
        # JAXopt expects bounds as a kwarg in the run method
        y_opt, state = solver.run(y0, args, bounds=bounds, **kwargs)

        payload = MinimizerPayload(y=y_opt)
        return payload, state