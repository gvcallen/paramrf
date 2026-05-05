"""
Optimistix optimization wrappers.
"""

from typing import Callable, Any

import equinox as eqx
from jaxtyping import PyTree
import optimistix as optx

from pmrf.optimize.base import AbstractUnconstrainedMinimizer, MinimizerPayload

class OptimistixMinimise(AbstractUnconstrainedMinimizer):
    """
    An optimizer that wraps :func:`optimistix.minimise`.
    """
    # Changed from AbstractLBFGS to the concrete BFGS solver
    solver: optx.AbstractMinimiser = eqx.field(default_factory=lambda: optx.BFGS(rtol=1e-6, atol=1e-6))

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizerPayload, PyTree]:
        
        result = optx.minimise(
            fn=fn, 
            solver=self.solver, 
            y0=y0, 
            args=args,
            max_steps=max_iter, 
            **kwargs
        )

        payload = MinimizerPayload(y=result.value)
        return payload, result