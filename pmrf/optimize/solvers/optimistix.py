"""
Optimistix optimization wrappers.
"""

from typing import Callable, Any

import jax.numpy as jnp
from jaxtyping import PyTree, Scalar
import optimistix as optx

from pmrf.optimize.base import AbstractUnconstrainedMinimizer, MinimizeResult


class OptimistixMinimise(AbstractUnconstrainedMinimizer):
    """
    An optimizer that wraps an arbitrary Optimistix :class:`optimistix.AbstractMinimiser`.
    
    Adds a function success tolerance to the solver, and also
    passes `throw=False` by default to :func:`optimistix.minimise`.
    
    Parameters
    ----------
    solver : optx.AbstractMinimiser
        The specific optimistix solver instance to use.
    fatol : float, default=1e-7
        Absolute tolerance of the function value for termination.
    """
    solver: optx.AbstractMinimiser
    fatol: float = 1e-7

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any,
        max_iter: int,
        **kwargs
    ) -> MinimizeResult:
        kwargs.setdefault('throw', False)
        
        result = optx.minimise(
            fn=fn, 
            solver=self.solver, 
            y0=y0, 
            args=args,
            max_steps=max_iter, 
            **kwargs
        )

        is_optx_success = (result.state == optx.RESULTS.successful)
        f_val = fn(result.value, args)
        f_converged = jnp.less_equal(jnp.abs(f_val), self.fatol)
        success = is_optx_success | f_converged

        return MinimizeResult(y=result.value, success=success, metrics=result)
      

class GradientDescent(AbstractUnconstrainedMinimizer):
    """
    A Gradient Descent optimizer in JAX.
    
    Wrapper around :class:`optimistix.GradientDescent`.

    Parameters
    ----------
    learning_rate : float
        Step size for the gradient descent updates.
    fatol : float, default=1e-7
        Absolute tolerance of the function value for termination.        
    step_atol : float, default=1e-6
        Absolute tolerance of the gradients and step sizes for termination.
    step_rtol : float, default=1e-6
        Relative tolerance of the gradients and step sizes for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    """
    learning_rate: float
    fatol: float = 1e-7
    step_atol: float = 1e-6
    step_rtol: float = 1e-6
    norm: Callable[[PyTree], Scalar] = optx.max_norm

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any,
        max_iter: int,
        **kwargs
    ) -> MinimizeResult:
        
        solver = optx.GradientDescent(
            learning_rate=self.learning_rate,
            rtol=self.step_rtol,
            atol=self.step_atol,
            norm=self.norm,
        )
        return OptimistixMinimise(solver, fatol=self.fatol).run(fn, y0, args, max_iter, **kwargs)
        

class BFGS(AbstractUnconstrainedMinimizer):
    """
    A BFGS optimizer in JAX.
    
    Wrapper around :class:`optimistix.BFGS`.

    Parameters
    ----------
    fatol : float, default=1e-7
        Absolute tolerance of the function value for termination.
    step_atol : float, default=1e-6
        Absolute tolerance of the gradients and step sizes for termination.
    step_rtol : float, default=1e-6
        Relative tolerance of the gradients and step sizes for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    use_inverse : bool, default=True
        Whether to use the inverse Hessian approximation.
    """
    fatol: float = 1e-7
    step_atol: float = 1e-6
    step_rtol: float = 1e-6
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    use_inverse: bool = True

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any,
        max_iter: int,
        **kwargs
    ) -> MinimizeResult:
        solver = optx.BFGS(
            rtol=self.step_rtol,
            atol=self.step_atol,
            norm=self.norm,
            use_inverse=self.use_inverse,
        )
        return OptimistixMinimise(solver, fatol=self.fatol).run(fn, y0, args, max_iter, **kwargs)
    
    
class LBFGS(AbstractUnconstrainedMinimizer):
    """
    A LBFGS optimizer in JAX.
    
    Wrapper around :class:`optimistix.LBFGS`.

    Parameters
    ----------
    fatol : float, default=1e-7
        Absolute tolerance of the function value for termination.
    step_atol : float, default=1e-6
        Absolute tolerance of the gradients and step sizes for termination.
    step_rtol : float, default=1e-6
        Relative tolerance of the gradients and step sizes for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    use_inverse : bool, default=True
        Whether to use the inverse Hessian approximation.
    """
    fatol: float = 1e-7
    step_atol: float = 1e-6
    step_rtol: float = 1e-6
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    use_inverse: bool = True

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any,
        max_iter: int,
        **kwargs
    ) -> MinimizeResult:
        solver = optx.LBFGS(
            rtol=self.step_rtol,
            atol=self.step_atol,
            norm=self.norm,
            use_inverse=self.use_inverse,
        )
        
        return OptimistixMinimise(solver, fatol=self.fatol).run(fn, y0, args, max_iter, **kwargs)

    
class NelderMead(AbstractUnconstrainedMinimizer):
    """
    A Nelder-Mead optimizer in JAX.
    
    Wrapper around :class:`optimistix.NelderMead`.

    Parameters
    ----------
    fatol : float, default=1e-7
        Absolute tolerance of the function value for termination.
    xatol : float, default=1e-6
        Absolute tolerance of the simplex for termination.
    xrtol : float, default=1e-6
        Relative tolerance of the simplex for termination.
    norm : Callable[[PyTree], Scalar], default=optx.max_norm
        Norm function used to evaluate the error.
    rdelta : float, default=5e-2
        Relative delta for the initial simplex.
    adelta : float, default=2.5e-4
        Absolute delta for the initial simplex.
    """
    fatol: float = 1e-7
    xatol: float = 1e-6
    xrtol: float = 1e-6
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    rdelta: float = 5e-2
    adelta: float = 2.5e-4

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any,
        max_iter: int,
        **kwargs
    ) -> MinimizeResult:
        
        solver = optx.NelderMead(
            rtol=self.xrtol,
            atol=self.xatol,
            norm=self.norm,
            rdelta=self.rdelta,
            adelta=self.adelta,
        )
        
        return OptimistixMinimise(solver, fatol=self.fatol).run(fn, y0, args, max_iter, **kwargs)