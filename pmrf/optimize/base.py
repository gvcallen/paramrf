"""
Base optimization functions and classes.
"""
import warnings
from typing import Any, Callable, TypeAlias
import abc

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar
import equinox as eqx
from pmrf._raw_space import RawSpace


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
    **kwargs
) -> tuple[PyTree, MinimizeResult]:
    """
    Optimizes the free parameters of a model, or any collection of models and parameters.

    The solver can be any solver of type `pmrf.optimize.AbstractMinimizer`. It moves
    through the free parameters in raw space, where constraints are enforced by each
    parameter's bijector, so a bounded solver is given infinite bounds. The solver
    receives a name-keyed dict, as from
    ``prf.param_values(model, free_only=True, space='raw')``; a solver that needs a
    1-D vector flattens it itself. The result is written back with
    ``prf.update(model, y, space='raw')``, so fixed parameters, names, scales and
    priors are unchanged.

    A parameter starting exactly on one of its bounds has an infinite raw value and
    could not move, so it raises; start it inside its bounds.

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
        If `model` has no free parameters, or one starts on a bound.
    """
    raw = RawSpace(model, 'optimize')
    if isinstance(solver, AbstractBoundedMinimizer):
        # Raw space is the whole real line; constraints are kept by the bijectors.
        kwargs['bounds'] = (
            jax.tree.map(lambda x: jnp.full_like(x, -jnp.inf), raw.y0),
            jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), raw.y0),
        )
    result = solver.run(fn=raw.objective(fn), y0=raw.y0, args=args, max_iter=max_iter, **kwargs)
    opt_model = raw.updated(result.y)

    if not result.success:
        warnings.warn("Optimization failed to converge. Try increasing the maximum number of iterations or loosening the solver tolerances.")

    return opt_model, result
