"""
Base optimization functions and classes.
"""
from typing import Any, Callable
import abc

import jax.numpy as jnp
from jaxtyping import PyTree, Scalar
import equinox as eqx
import parax as prx

from pmrf.core import Model, Frequency


class MinimizerPayload(eqx.Module):
    """The core mathematical payload of a minimization run."""
    #: The optimal arrays (y_opt)
    y: PyTree


class AbstractUnconstrainedMinimizer(eqx.Module):
    """
    An interface for JAX-wrapped minimization algorithms that require a single call.

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
    An interface for JAX-wrapped minimization algorithms
    that cater for bounds.

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


class OptimizeResult(prx.Module):
    """
    The result of an optimization run.
    """
    #: The RF model holding the final optimized parameters.
    model: Model

    #: The objective function (e.g., :class:`pmrf.evaluators.TargetLoss`)
    #: used to calculate the objective during optimization. If the objective was a module
    #: with hyper-parameters, then this contains the optimized objective model.
    objective: Callable[[Model, Frequency], jnp.ndarray]

    #: The final objective function value achieved by the optimizer.
    objective_value: jnp.ndarray
    
    #: The underlying results object returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    #: Not saved to file.
    solver_results: Any = prx.constrained(default=None, save=False)