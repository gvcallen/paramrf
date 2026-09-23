"""
The view of a model a solver moves through, shared by the minimiser and every sampler.

A solver never sees the model itself: it sees the free parameters as a name-keyed dict
of values, in raw space for the minimiser and the joint and split samplers, and in
declared space for the hypercube samplers. Everything here goes through the public
:func:`pmrf.values`, :func:`pmrf.update` and :func:`pmrf.log_prior`, so a solver
sees exactly what a user would with those functions.
"""

from typing import Any, Callable

from jaxtyping import PyTree, Scalar

import jax
import jax.numpy as jnp

from pmrf.parameters import log_prior as _log_prior, values as _values, update
from pmrf.utils import unwrap


class SolverView:
    """
    The free parameters of a model, as the dict of values a solver moves through.

    The starting values :attr:`y0` are raw, and :meth:`updated` and :meth:`objective`
    take a ``space`` so a solver working in declared space, such as a hypercube
    sampler, can use the same view.

    Parameters
    ----------
    model : PyTree
        A model, or any collection of models and parameters.
    action : str
        What the solver does, such as ``'optimize'``, for the error raised when there
        is nothing to move.

    Raises
    ------
    ValueError
        If `model` has no free parameters.
    """

    def __init__(self, model: PyTree, action: str):
        #: The model the values are written into.
        self.model = model
        #: What the solver does, used in error messages.
        self.action = action
        #: The starting raw values of the free parameters, by name.
        self.y0 = _values(model, free_only=True, space='raw')
        if not self.y0:
            raise ValueError(
                f"Nothing to {action}: the tree has no free parameters. Every parameter is "
                "either fixed or a plain value."
            )

    def check_finite(self) -> None:
        """Checks that every starting raw value is finite, so a solver can move it.

        Only for solvers that move raw values: a parameter exactly on a bound has an
        infinite raw value, which is fine for a solver working in declared space.

        Raises
        ------
        ValueError
            If a starting raw value is NaN, or infinite because the parameter starts
            on one of its bounds.
        """
        def _leaves(v):
            return [jnp.asarray(x) for x in jax.tree.leaves(v)]

        nan = [name for name, v in self.y0.items() if any(bool(jnp.any(jnp.isnan(x))) for x in _leaves(v))]
        if nan:
            raise ValueError(
                f"Cannot {self.action}: {', '.join(repr(name) for name in nan)} start at NaN, "
                "which no solver can move. Give them a finite starting value."
            )
        on_bound = [name for name, v in self.y0.items() if any(bool(jnp.any(jnp.isinf(x))) for x in _leaves(v))]
        if on_bound:
            raise ValueError(
                f"Cannot {self.action}: {', '.join(repr(name) for name in on_bound)} start on a bound, "
                "where the raw value is infinite and cannot move. Start them inside their bounds."
            )

    def updated(self, values: dict, space: str = 'raw') -> PyTree:
        """Returns the model with `values`, by name and in `space`, written into it.

        Batched values give a batched model; fixed parameters stay unbatched.
        """
        return update(self.model, values, space=space)

    def read(self, batch: PyTree) -> dict:
        """Returns the raw values of the free parameters of `batch`, a model like this one.

        Raises
        ------
        ValueError
            If a free parameter of this model is missing from `batch`, or is fixed there.
        """
        values = _values(batch, free_only=True, space='raw')
        missing = [name for name in self.y0 if name not in values]
        if missing:
            raise ValueError(
                f"Cannot {self.action}: {', '.join(repr(name) for name in missing)} "
                f"{'is' if len(missing) == 1 else 'are'} free in the model but missing or "
                "fixed in the given tree. It must have the same free parameters as the model."
            )
        return {name: values[name] for name in self.y0}

    def objective(self, fn: Callable[[PyTree, Any], Scalar], space: str = 'raw') -> Callable[[dict, Any], Scalar]:
        """Returns `fn`, which takes the unwrapped model, as a function of values in `space`."""
        def value_fn(values: dict, args: Any) -> Scalar:
            return fn(unwrap(self.updated(values, space)), args)
        return value_fn

    def log_prior(self, values: dict, _args: Any = None) -> Scalar:
        """Returns the raw-space log prior of the model at `values`."""
        return _log_prior(self.updated(values), space='raw')

    def log_posterior(self, loglikelihood_fn: Callable[[PyTree, Any], Scalar]) -> Callable[[dict, Any], Scalar]:
        """Returns the raw-space log posterior: `loglikelihood_fn` plus the raw log prior."""
        def value_fn(values: dict, args: Any) -> Scalar:
            model = self.updated(values)
            return loglikelihood_fn(unwrap(model), args) + _log_prior(model, space='raw')
        return value_fn
