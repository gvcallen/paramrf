"""
Base optimization functions and classes.
"""
from typing import Any, Callable
import abc

import jax.numpy as jnp
from jaxtyping import PyTree
import optimistix as optx
import equinox as eqx
import parax as prx

from pmrf.core import Model, Frequency


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


class AbstractCallableMinimizer(eqx.Module):
    """
    An interface for JAX-wrapped minimization algorithms that require a single `__call__`.

    Provided to cater for algorithms that `Optimistix` does not support.

    The interface should accept pure PyTrees and return a standardized `optx.Solution`.
    """
    #: Signifies whether the minimizer supports bounds or not.
    #: If True, PyTree bounds will be passed in options['lower'] and options['upper'].
    supports_bounds: eqx.AbstractClassVar[bool]

    @abc.abstractmethod
    def __call__(
        self,
        fn: Callable[[PyTree], PyTree],
        y0: PyTree,
        args: PyTree[Any],
        options: dict[str, Any],
        max_steps: int = 1024,
        **kwargs
    ) -> optx.Solution:
        """
        Execute the minimization algorithm.

        Parameters
        ----------
        fn : callable
            The objective function to minimize.
        y0 : PyTree
            The initial parameter guess.
        args : PyTree
            Additional static arguments passed to the objective function.
        options : dict
            Runtime configuration for the solver. If `supports_bounds` is True,
            boundary constraints are provided here via 'lower' and 'upper' keys.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        optx.Solution
            A standard optimistix solution object containing the optimized parameters,
            convergence status, and solver statistics.
        """
        raise NotImplementedError
    
def is_minimizer(x):
    """
    Returns if a solver is suitable for minimization in :mod:`pmrf.optimize.minimize`.

    Returns `True` for :class:`pmrf.optimize.ScipyMinimize` and :class:`optimistix.AbstractMinimiser`.
    """
    return isinstance(x, AbstractCallableMinimizer | optx.AbstractMinimiser)

def is_optimizer(x):
    """
    Returns if a solver is suitable for frequentist optimization in :mod:`pmrf.optimize`.

    Returns `True` for :class:`pmrf.optimize.ScipyMinimize` and :class:`optimistix.AbstractMinimiser`.
    """
    return is_minimizer(x)
