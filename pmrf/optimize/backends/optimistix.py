"""
Optimistix optimization wrappers.
"""

from typing import Callable, Any

import equinox as eqx
from jaxtyping import PyTree, Scalar
import optimistix as optx

from pmrf.optimize.base import AbstractUnconstrainedMinimizer, MinimizeResult

DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-3

class OptimistixMinimise(AbstractUnconstrainedMinimizer):
    """
    An optimizer that wraps :func:`optimistix.minimise`.

    Parameters
    ----------
    solver : optx.AbstractMinimiser
        The specific optimistix solver instance to use.
    """
    # Changed from AbstractLBFGS to the concrete BFGS solver
    solver: optx.AbstractMinimiser

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizeResult, PyTree]:
        
        result = optx.minimise(
            fn=fn, 
            solver=self.solver, 
            y0=y0, 
            args=args,
            max_steps=max_iter, 
            **kwargs
        )

        payload = MinimizeResult(y=result.value, success=(result.state == optx.RESULTS.successful))
        return payload, result
    

class NelderMead(AbstractUnconstrainedMinimizer):
    """
    An optimizer that wraps optimistix's Nelder-Mead.

    Parameters
    ----------
    rtol : float, default=1e-6
        Relative tolerance for termination.
    atol : float, default=1e-1
        Absolute tolerance for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    rdelta : float, default=5e-2
        Relative delta for the initial simplex.
    adelta : float, default=2.5e-4
        Absolute delta for the initial simplex.
    """
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    rdelta: float = 5e-2
    adelta: float = 2.5e-4

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizeResult, PyTree]:
        
        solver = optx.NelderMead(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            rdelta=self.rdelta,
            adelta=self.adelta,
        )
        
        result = optx.minimise(
            fn=fn, 
            solver=solver,
            y0=y0, 
            args=args,
            max_steps=max_iter, 
            throw=False,
            **kwargs
        )

        payload = MinimizeResult(y=result.value, success=(result.state == optx.RESULTS.successful))
        return payload, result
    

class GradientDescent(AbstractUnconstrainedMinimizer):
    """
    An optimizer that wraps optimistix's Gradient Descent.

    Parameters
    ----------
    learning_rate : float
        Step size for the gradient descent updates.
    rtol : float, default=1e-6
        Relative tolerance for termination.
    atol : float, default=1e-1
        Absolute tolerance for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    """
    learning_rate: float
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    norm: Callable[[PyTree], Scalar] = optx.max_norm

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizeResult, PyTree]:
        
        solver = optx.GradientDescent(
            learning_rate=self.learning_rate,
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
        )
        
        result = optx.minimise(
            fn=fn, 
            solver=solver,
            y0=y0, 
            args=args,
            max_steps=max_iter,
            throw=False,
            **kwargs
        )
        
        payload = MinimizeResult(y=result.value, success=(result.state == optx.RESULTS.successful))
        return payload, result
    

class LBFGS(AbstractUnconstrainedMinimizer):
    """
    An optimizer that wraps optimistix's LBFGS.

    Parameters
    ----------
    rtol : float, default=1e-6
        Relative tolerance for termination.
    atol : float, default=1e-1
        Absolute tolerance for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    use_inverse : bool, default=True
        Whether to use the inverse Hessian approximation.
    """
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    use_inverse: bool = True,

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizeResult, PyTree]:
        solver = optx.LBFGS(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            use_inverse=self.use_inverse,
        )
        
        result = optx.minimise(
            fn=fn, 
            solver=solver,
            y0=y0, 
            args=args,
            max_steps=max_iter,
            throw=False,
            **kwargs
        )

        payload = MinimizeResult(y=result.value, success=(result.state == optx.RESULTS.successful))
        return payload, result
    

class BFGS(AbstractUnconstrainedMinimizer):
    """
    An optimizer that wraps optimistix's BFGS.

    Parameters
    ----------
    rtol : float, default=1e-6
        Relative tolerance for termination.
    atol : float, default=1e-1
        Absolute tolerance for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    use_inverse : bool, default=True
        Whether to use the inverse Hessian approximation.
    """
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    use_inverse: bool = True,

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizeResult, PyTree]:
        solver = optx.BFGS(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            use_inverse=self.use_inverse,
        )

        result = optx.minimise(
            fn=fn, 
            solver=solver,
            y0=y0, 
            args=args,
            max_steps=max_iter,
            throw=False,
            **kwargs
        )

        payload = MinimizeResult(y=result.value, success=(result.state == optx.RESULTS.successful))
        return payload, result