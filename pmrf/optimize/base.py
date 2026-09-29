"""
Base optimization functions and classes.
"""
import warnings
from typing import Any, Callable, TypeAlias
import abc

from jaxtyping import PyTree, Scalar
import equinox as eqx
from pmrf._solver_view import SolverView


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

    It does not honour bounds, so it moves through raw space, where each parameter's
    bijector keeps it inside its bounds.
    """

    @property
    def honours_bounds(self) -> bool:
        """Whether the minimiser keeps its values inside the bounds it is given: never."""
        return False

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

    A minimiser that honours bounds (:attr:`honours_bounds`) searches box space, the
    box of the parameters' bounds, and is given its edges as `bounds`. One that does
    not, such as a backend configured with an unbounded method, moves through raw
    space instead and is given no `bounds`.
    """

    @property
    def honours_bounds(self) -> bool:
        """Whether the minimiser keeps its values inside the bounds it is given.

        True by default. A subclass whose backend can ignore bounds overrides it.
        """
        return True

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
        bounds : tuple[PyTree, PyTree], optional
            The lower and upper edges of the box to search, shaped like `y0`, or None
            if the minimiser does not honour bounds. Edges may be infinite.
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
    **kwargs
) -> tuple[PyTree, MinimizeResult]:
    """
    Optimizes the free parameters of a model, or any collection of models and parameters.

    The solver can be any solver of type `pmrf.optimize.AbstractMinimizer`. It receives
    the free parameters as a name-keyed dict; a solver that needs a 1-D vector flattens
    it itself. The space depends on whether the solver honours bounds
    (``solver.honours_bounds``):

    - **Box space**, if it does. Each parameter is searched over the box of its bounds,
      not its prior: the unit box between two finite bounds, declared space otherwise.
      The solver is given the box's edges as `bounds`. A closed edge is the bound
      itself, so a parameter can start on it, reach it and stay on it; an open edge is
      inset by 1e-6, so it is never evaluated.
    - **Raw space**, if it does not. Constraints are enforced by each parameter's
      bijector. A parameter starting exactly on a closed bound has an infinite raw
      value, so the solver starts it just inside: by 1e-6 of the width between two
      finite bounds, or 1e-6 in declared units otherwise.

    The result is written back with :func:`pmrf.update`, so fixed parameters, names,
    scales and priors are unchanged.

    Parameters
    ----------
    fn : callable
        The objective function taking `(unwrapped_model, args)`.
    model : PyTree
        The initial model.
    solver : AbstractMinimizer
        The instantiated optimizer to run.
    args : Any, optional
        Args to pass to `fn`. Defaults to None.
    max_iter : int, optional
        Maximum number of iterations. Defaults to 1024.
    **kwargs
        Runtime arguments forwarded to the solver backend.

    Returns
    -------
    tuple
        A tuple of `(best_model, minimize_results)`.

    Raises
    ------
    ValueError
        If `model` has no free parameters, or one starts at NaN.
    """
    view = SolverView(model, 'optimize')
    # A start at NaN cannot move in either space. A start on a closed bound is already
    # nudged inside it in raw space.
    view.check_finite()
    if solver.honours_bounds:
        space = 'box'
        y0, kwargs['bounds'] = view.box()
    else:
        space, y0 = 'raw', view.y0
    result = solver.run(fn=view.objective(fn, space), y0=y0, args=args, max_iter=max_iter, **kwargs)
    opt_model = view.updated(result.y, space)

    if not result.success:
        warnings.warn("Optimization failed to converge. Try increasing the maximum number of iterations or loosening the solver tolerances.")

    return opt_model, result
