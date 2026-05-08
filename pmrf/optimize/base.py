"""
Base optimization functions and classes.
"""
from typing import Any, Callable
import abc

from jaxtyping import PyTree, Scalar
import equinox as eqx
import parax as prx
from pmrf.jax_utils import partition


class MinimizerPayload(eqx.Module):
    """The core mathematical payload of a minimization run."""
    #: The optimal arrays (y_opt)
    y: PyTree


class AbstractUnconstrainedMinimizer(eqx.Module):
    """
    Abstract interface for unconstrained minimization.

    The interface should accept pure PyTrees and return a standardized tuple.
    """

    @abc.abstractmethod
    def run(
        self,
        fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: Any = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizerPayload, PyTree]:
        """
        Execute the minimization algorithm.

        Parameters
        ----------
        fn : callable
            The objective function to minimize.
        y0 : PyTree
            The initial parameter guess.
        args : Any
            Args to pass to `fn`.            
        max_iter: int = 1024
            The maximum number of iterations to take.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        tuple
            A tuple of `(MinimizerPayload, metrics)`.
        """
        raise NotImplementedError
    

class AbstractBoundedMinimizer(eqx.Module):
    """
    Abstract interface for bounded minimization.


    The interface should accept pure PyTrees and return a standardized tuple.
    """

    @abc.abstractmethod
    def run(
        self,
        fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: Any = None,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> tuple[MinimizerPayload, PyTree]:
        """
        Execute the minimization algorithm.

        Parameters
        ----------
        fn : callable
            The objective function to minimize.
        y0 : PyTree
            The initial parameter guess.
        args : Any
            Args to pass to `fn`.
        bounds : PyTree
            Bounds for `y0`, if any.
        max_iter: int = 1024
            The maximum number of iterations to take.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        tuple
            A tuple of `(MinimizerPayload, metrics)`.
        """
        raise NotImplementedError
    

AbstractMinimizer = AbstractUnconstrainedMinimizer | AbstractBoundedMinimizer
    

def is_minimizer(x):
    """
    Returns True if x is an instance of `pmrf.optimize.AbstractUnconstrainedMinimizer`
    or `pmrf.optimize.AbstractBoundedMinimizer`.
    """
    return isinstance(x, AbstractMinimizer)


def is_optimizer(x):
    """Returns True if `pmrf.optimize.is_minimizer` returns True."""
    return is_minimizer(x)


def minimize(
    fn: Callable[[PyTree, Any], Scalar], 
    y0: PyTree, 
    solver: AbstractMinimizer,
    args: Any = None,
    max_iter: int = 1024, 
    **kwargs
) -> tuple[PyTree, MinimizerPayload, PyTree]:
    """
    Optimizes a general PyTree potentially containing Parax parameters using either a bounded or unconstrained solver.
    """
    is_bounded = isinstance(solver, AbstractBoundedMinimizer)
    
    # Extract base values and partition based on solver type
    if is_bounded:
        base_tree = prx.unwrap(y0, only_if=prx.is_bounded)
        lower_bounds, upper_bounds = prx.bounded.tree_bounds(y0)
        
        params, static = partition(base_tree)
        lower, _ = partition(lower_bounds)
        upper, _ = partition(upper_bounds)
        bounds = (lower, upper)
    else:
        params, static = partition(y0)

    # Define the unified objective wrapper for the solver
    def objective(p: PyTree, args: Any) -> Scalar:
        unwrapped_model = prx.unwrap(eqx.combine(p, static))
        return fn(unwrapped_model, args)

    # Run the correct solver execution and reconstruct the final model
    if is_bounded:
        payload, metrics = solver.run(
            fn=objective, y0=params, args=args, bounds=bounds, max_iter=max_iter, **kwargs
        )
        opt_base = eqx.combine(payload.y, static)
        final_model = prx.wrap(y0, opt_base, only_if=prx.is_bounded)
    else:
        payload, metrics = solver.run(
            fn=objective, y0=params, args=args, max_iter=max_iter, **kwargs
        )
        final_model = eqx.combine(payload.y, static)

    return final_model, payload, metrics