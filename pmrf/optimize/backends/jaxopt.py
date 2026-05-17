"""
JAXopt optimization wrappers.
"""

from typing import Callable, Any

import equinox as eqx
from jaxtyping import PyTree
import jaxopt

from pmrf.optimize.base import AbstractBoundedMinimizer, MinimizeResult

class LBFGSB(AbstractBoundedMinimizer):
    """
    A L-BFGS-B optimizer in JAX.
    
    Wrapper around :class:`jaxopt.LBFGSB`.
    
    Parameters
    ----------
    gtol : float, default=1e-3
        The gradient norm tolerance for termination.
    stepsize : float, default=1.0
        Initial step size for the line search.
    linesearch : str, default="zoom"
        Type of line search to use.
    """
    gtol: float = eqx.field(static=True, default=1e-3)
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
    ) -> tuple[MinimizeResult, PyTree]:
        solver = jaxopt.LBFGSB(
            fun=fn,
            tol=self.gtol,
            stepsize=self.stepsize,
            maxiter=max_iter,
            linesearch=self.linesearch,
        )
        
        # JAXopt expects bounds as a kwarg in the run method
        y_opt, state = solver.run(y0, bounds, args, **kwargs)

        converged = state.error <= self.gtol
        payload = MinimizeResult(y=y_opt, success=converged)
        return payload, state