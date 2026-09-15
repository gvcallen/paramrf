"""
Base optimization functions and classes.
"""
import warnings
from typing import Any, Callable, TypeAlias
import abc

from jaxtyping import PyTree, Scalar
import equinox as eqx

from pmrf.parameters import flatten, is_leaf


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

    flat = flatten(model, space="physical")
    if not flat.names:
        raise ValueError(
            "Nothing to optimize: the tree has no free parameters. Every parameter is "
            "either fixed or a plain value."
        )

    # The solver works on the unscaled constrained values, in the constraints' base
    # space when bounded and on the real line otherwise.
    if is_bounded:
        bijector = flat._constraint.base_bijector
        bounds = flat._constraint.base_bounds
    else:
        bijector = flat._constraint.bijector
        bounds = None

    def to_physical(p: PyTree) -> PyTree:
        return flat._scale(bijector.forward(p))

    solver_params = bijector.inverse(flat._scale(flat._unravel(flat.theta0), inverse=True))

    def objective(p: PyTree, args: Any) -> Scalar:
        return fn(flat._unflatten_tree(to_physical(p)), args)

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
    opt_model = eqx.combine(flat._static, flat._wrap_tree(to_physical(result.y)), is_leaf=is_leaf)
        
    if not result.success:
        warnings.warn("Optimization failed to converge. Try increasing the maximum number of iterations or loosening the solver tolerances.")

    return opt_model, result