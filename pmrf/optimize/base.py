"""
Base optimization functions and classes.
"""
import warnings
from typing import Any, Callable, TypeAlias
import abc

import jax
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

    The solver can be any solver of type `pmrf.optimize.AbstractMinimizer`.

    Performs Equinox partitioning and Parax unwrapping/extraction,
    as well as delegation to the relevant solver interface. Bounded solvers
    are automatically routed to operate within the orthogonal base bounds 
    of the parameter constraints, protecting them from spatial correlations.

    Note that all Parax unwrappables (such as Parax variables)
    MUST be re-wrappable for this interface.

    Parameters
    ----------
    fn : callable
        The objective function taking `(unwrapped_y0, args)`.
    model : PyTree
        The initial parameter guess / model state.
    solver : AbstractMinimizer
        The instantiated optimizer to run.
    args : Any, optional
        Args to pass to `fn`. Defaults to None.
    max_iter : int, optional
        Maximum number of iterations. Defaults to 1024.
    use_bounds : bool, optional
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

    # Unified Equinox partitioning using the constraints API
    is_dynamic = lambda x: prx.constraints.is_dynamic(x) and not isinstance(x, np.ndarray)
    is_leaf = prx.constraints.is_leaf
    
    dynamic, static = eqx.partition(model, is_dynamic, is_leaf=is_leaf)
    if not jax.tree.leaves(dynamic):
        raise ValueError(
            "Nothing to optimize: the tree has no free parameters. Every parameter is "
            "either fixed or a plain value."
        )
    physical_params = prx.unwrap(dynamic, only_if=prx.is_constrained)

    # Configure the spatial projection and bounds based on the solver type
    leafwise_constraint = prx.constraints.tree_leafwise_constraint(dynamic)
    if is_bounded:
        bijector = leafwise_constraint.base_bijector
        bounds = leafwise_constraint.base_bounds
    else:
        bijector = leafwise_constraint.bijector
        bounds = None

    # Map the physical starting parameters into the solver's operational space
    solver_params = bijector.inverse(physical_params)

    # The objective always projects the solver's parameters forward to physical space.
    # `static` is passed first here so it drives `eqx.combine`'s structural matching
    def objective(p: PyTree, args: Any) -> Scalar:
        physical_p = bijector.forward(p)
        unwrapped_model = prx.unwrap(eqx.combine(static, physical_p, is_leaf=is_leaf))
        return fn(unwrapped_model, args)

    # Execute the backend solver
    if is_bounded:
        result = solver.run(
            fn=objective, y0=solver_params, args=args, bounds=bounds, max_iter=max_iter, **kwargs
        )
    else:
        result = solver.run(
            fn=objective, y0=solver_params, args=args, max_iter=max_iter, **kwargs
        )

    # Re-wrap the optimized parameters into the constrained physical domain
    opt_physical = bijector.forward(result.y)
    opt_dynamic = prx.wrap(dynamic, opt_physical, only_if=prx.is_constrained)
    opt_model = eqx.combine(opt_dynamic, static, is_leaf=is_leaf)
        
    if not result.success:
        warnings.warn("Optimization failed to converge. Try increasing the maximum number of iterations or loosening the solver tolerances.")

    return opt_model, result