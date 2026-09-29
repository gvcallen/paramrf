"""
The view of a model a solver moves through, shared by the minimiser and every sampler.

A solver never sees the model itself: it sees the free parameters as a name-keyed dict
of values, in raw space for the minimiser and the joint and split samplers, and in
declared space for the hypercube samplers. Everything here goes through the public
:func:`pmrf.values`, :func:`pmrf.update` and :func:`pmrf.log_prior`, so a solver
sees exactly what a user would with those functions.

Box space is the box a bounded minimiser searches (ADR-0007): the unit box along an
element with two finite bounds, and declared space along any other.
"""

from typing import Any, Callable

from jaxtyping import Array, ArrayLike, PyTree, Scalar

import jax
import jax.numpy as jnp
import parax as prx

from pmrf.parameters import Param, is_param, log_prior as _log_prior, params, values as _values, update
from pmrf.utils import unwrap


#: How far a start on a closed bound is moved inward, in box space (ADR-0007).
NUDGE = 1e-6


def _box_frame(lower: Array, upper: Array) -> tuple[Array, Array]:
    """Returns the origin and width that map declared values into box space."""
    finite = jnp.isfinite(lower) & jnp.isfinite(upper)
    return jnp.where(finite, lower, 0.0), jnp.where(finite, upper - lower, 1.0)


def to_box(value: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> Array:
    """Returns the declared `value` in the box space of the bounds `lower` and `upper`."""
    origin, width = _box_frame(jnp.asarray(lower), jnp.asarray(upper))
    return (jnp.asarray(value) - origin) / width


def from_box(value: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> Array:
    """Returns the box-space `value` in the declared space of the bounds `lower` and `upper`."""
    origin, width = _box_frame(jnp.asarray(lower), jnp.asarray(upper))
    return origin + jnp.asarray(value) * width


def _nudged(param: Param) -> Array | None:
    """Returns the declared value of `param` with each element on a closed bound moved
    :data:`NUDGE` inward in box space, or None if no element is on one.

    An element also counts as on a bound when it is no further from it than twice the
    image of that bound through the raw-to-declared bijector. A Parax prior whitening
    clips on the way out of raw space, so a prior's value set on its bound reads back
    slightly inside it, with a raw value too far out to move.
    """
    constraint = prx.unwrap(param.constraint)
    if constraint is None:
        return None
    value = param.value

    def _like_value(x):
        return jnp.broadcast_to(jnp.asarray(x), value.shape)

    lower, upper = (_like_value(b).astype(value.dtype) for b in constraint.bounds)
    lower_closed, upper_closed = (_like_value(c) for c in constraint.closed)
    ends = [_like_value(constraint.bijector.forward(jnp.full(value.shape, x, value.dtype))) for x in (-jnp.inf, jnp.inf)]
    lower_image = jnp.where(jnp.isnan(ends[0]) | jnp.isnan(ends[1]), lower, jnp.minimum(*ends))
    upper_image = jnp.where(jnp.isnan(ends[0]) | jnp.isnan(ends[1]), upper, jnp.maximum(*ends))

    at_lower = lower_closed & jnp.isfinite(lower) & (value - lower <= 2 * (lower_image - lower))
    at_upper = upper_closed & jnp.isfinite(upper) & ~at_lower & (upper - value <= 2 * (upper - upper_image))
    if not bool(jnp.any(at_lower | at_upper)):
        return None

    box = jnp.where(at_lower, to_box(lower, lower, upper) + NUDGE, to_box(upper, lower, upper) - NUDGE)
    return jnp.where(at_lower | at_upper, from_box(box, lower, upper), value).astype(value.dtype)


class SolverView:
    """
    The free parameters of a model, as the dict of values a solver moves through.

    The starting values :attr:`y0` are raw, and :meth:`updated` and :meth:`objective`
    take a ``space`` so a solver working in declared space, such as a hypercube
    sampler, can use the same view.

    A parameter starting on a closed bound has an infinite raw value, so :attr:`y0`
    starts it :data:`NUDGE` inward in box space instead (ADR-0007). The model keeps
    the value it was given.

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
        raw = _values(model, free_only=True, space='raw')
        if not raw:
            raise ValueError(
                f"Nothing to {action}: the tree has no free parameters. Every parameter is "
                "either fixed or a plain value."
            )
        nudged = {}
        for name, param in params(model, free_only=True).items():
            if is_param(param) and (value := _nudged(param)) is not None:
                nudged[name] = value
        if nudged:
            raw = _values(update(model, nudged), free_only=True, space='raw')
        #: The starting raw values of the free parameters, by name, with each start on
        #: a closed bound nudged inward.
        self.y0 = raw

    def check_finite(self) -> None:
        """Checks that no starting raw value is NaN, so a solver can move it.

        Only for solvers that move raw values.

        Raises
        ------
        ValueError
            If a starting raw value is NaN.
        """
        def _leaves(v):
            return [jnp.asarray(x) for x in jax.tree.leaves(v)]

        nan = [name for name, v in self.y0.items() if any(bool(jnp.any(jnp.isnan(x))) for x in _leaves(v))]
        if nan:
            raise ValueError(
                f"Cannot {self.action}: {', '.join(repr(name) for name in nan)} start at NaN, "
                "which no solver can move. Give them a finite starting value."
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
