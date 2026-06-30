"""
Base optimization functions and classes.
"""
import warnings
from typing import Any, Callable, TypeAlias
import abc

import numpy as np
from jaxtyping import PyTree, Scalar
import equinox as eqx
import parax as prx


class MinimizeResult(eqx.Module):
    """The core mathematical payload of a minimization run."""
    #: The optimal arrays (y_opt)
    y: PyTree
    
    #: Whether the algorithm successfully converged
    success: bool = True
    
    #: Any solver metrics.
    metrics: Any = None


class AbstractUnconstrainedMinimizer(eqx.Module):
    """
    Abstract interface for unconstrained minimization algorithms.
    """

    @abc.abstractmethod
    def run(
        self,
        fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: Any,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
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
        results
            An instance of :class:`pmrf.optimize.MinimizeResult`.
        """
        raise NotImplementedError
    

class AbstractBoundedMinimizer(eqx.Module):
    """
    Abstract interface for bounded minimization algorithms.
    """

    @abc.abstractmethod
    def run(
        self,
        fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: Any,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
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
        results
            An instance of :class:`pmrf.optimize.MinimizeResult`.
        """
        raise NotImplementedError
    

#: A type-hint for a minimizer in :mod:`pmrf.optimize`. Either :class:`pmrf.optimize.AbstractUnconstrainedMinimizer` or :class:`pmrf.optimize.AbstractBoundedMinimizer`.
AbstractMinimizer: TypeAlias = AbstractUnconstrainedMinimizer | AbstractBoundedMinimizer
    

def is_minimizer(x):
    """
    Returns True if x is an instance of `pmrf.optimize.AbstractUnconstrainedMinimizer`
    or `pmrf.optimize.AbstractBoundedMinimizer`.
    """
    return isinstance(x, AbstractMinimizer)


def is_optimizer(x):
    """Returns True if `pmrf.optimize.is_minimizer` returns True."""
    return is_minimizer(x)


def run_minimizer(
    fn: Callable[[PyTree, Any], Scalar], 
    model: PyTree, 
    solver: AbstractMinimizer,
    args: Any = None,
    max_iter: int = 1024, 
    use_bounds: bool | None = None,
    **kwargs
) -> tuple[PyTree, MinimizeResult]:
    """
    Optimizes a general PyTree potentially containing Parax parameters.

    The solver can be any solver of type :type:`pmrf.optimize.AbstractMinimizer`.

    Performs Equinox partitioning and Parax unwrrapping/extraction,
    as well as delegation to the relevant solver interface.

    Note that all Parax unwrappables (such a Parax variables)
    MUST be re-wrappable for this interface.

    Parameters
    ----------
    fn : callable
        The likelihood function taking `(unwrapped_y0, args)`.
    model : PyTree
        The initial parameter guess / model state.
    solver : AbstractMinimizer
        The instantiated sampler to run.
    args : Any
        Args to pass to `fn`.
    max_iter: int, optional
        Maximum number of iterations.
    use_bounds: int, optional
        Whether bounds should be used. Defaults to True only if the solver is bounded.
    **kwargs
        Runtime arguments forwarded to the solver backend.

    Returns
    -------
    tuple
        A tuple of `(best_model, minimize_results)`.    
    """
    is_bounded = isinstance(solver, AbstractBoundedMinimizer)
    if use_bounds is not None:
        is_bounded = is_bounded and use_bounds

    # Extract base values and partition based on solver type
    if is_bounded:
        is_dynamic = lambda x: prx.bounds.is_dynamic(x) and not isinstance(x, np.ndarray)
        is_leaf = prx.bounds.is_leaf
        
        dynamic, static = eqx.partition(model, is_dynamic, is_leaf=is_leaf)
        
        params = prx.unwrap(dynamic, only_if=prx.is_bounded)
        bounds = prx.bounds.tree_bounds(dynamic)
    else:
        is_dynamic = lambda x: eqx.is_inexact_array(x) and not isinstance(x, np.ndarray)
        is_leaf = prx.is_constant
        
        params, static = eqx.partition(model, is_dynamic, is_leaf=is_leaf)

    # Define the unified objective wrapper for the solver
    def objective(p: PyTree, args: Any) -> Scalar:
        unwrapped_model = prx.unwrap(eqx.combine(p, static, is_leaf=is_leaf))
        return fn(unwrapped_model, args)

    # Run the correct solver execution and reconstruct the final model
    if is_bounded:
        result = solver.run(
            fn=objective, y0=params, args=args, bounds=bounds, max_iter=max_iter, **kwargs
        )
        # Re-wrap into unbounded domain
        opt_dynamic = prx.wrap(dynamic, result.y, only_if=prx.is_bounded)
        opt_model = eqx.combine(opt_dynamic, static, is_leaf=is_leaf)
    else:
        result = solver.run(
            fn=objective, y0=params, args=args, max_iter=max_iter, **kwargs
        )
        # No need to re-wrap because `params` was never unwrapped
        opt_model = eqx.combine(result.y, static, is_leaf=is_leaf)
        
    if not result.success:
        warnings.warn("Optimization failed to converge. Trying increasing the maximum number of iterations or loosening the solver tolerances.")

    return opt_model, result