"""
The view of a model a solver moves through, shared by the minimiser and every sampler.

A solver never sees the model itself: it sees the free parameters as a name-keyed dict
of values: in box space for a minimiser that honours bounds, in raw space for one that
does not and for the joint and split samplers, and in declared space for the hypercube
samplers. Values are read and written through the public :func:`pmrf.values`,
:func:`pmrf.update` and :func:`pmrf.log_prior`, so a solver sees exactly what a user
would with those functions.

Box space is the box a bounded minimiser searches (ADR-0007). Along each parameter it is
the constraint's Parax base space, built from its bounds and not its prior: the unit box
between two finite bounds, declared space otherwise. A closed edge of the box is the
bound itself; an open edge is inset by 1e-6.
"""

import dataclasses
from typing import Any, Callable

from jaxtyping import Array, PyTree, Scalar

import equinox as eqx
import jax
import jax.numpy as jnp
import parax as prx

from pmrf.parameters import (
    Param, is_param, log_prior as _log_prior, params, tree_param_paths, values as _values, update,
)
from pmrf.utils import unwrap
from pmrf.utils.tree import Pathgetter


# How far a start on a closed bound is moved inward, and an open edge of the box is
# inset, in box space (ADR-0007).
_NUDGE = 1e-6


def _box_constraint(node: Param | Array) -> prx.constraints.AbstractConstraint | None:
    """Returns the constraint whose base space is the box of `node`, or None if its box
    is its declared space: a raw array, or a parameter without bounds."""
    return prx.unwrap(node.constraint) if is_param(node) else None


def _box_edges(node: Param | Array) -> tuple[Array, Array]:
    """Returns the lower and upper edges of the box of `node`, shaped like its value."""
    value = jnp.asarray(node.value if is_param(node) else node)
    constraint = _box_constraint(node)
    if constraint is None:
        return jnp.full_like(value, -jnp.inf), jnp.full_like(value, jnp.inf)

    def _like_value(x):
        return jnp.broadcast_to(jnp.asarray(x), value.shape).astype(value.dtype)

    lower, upper = (_like_value(b) for b in constraint.base_bounds)
    lower_closed, upper_closed = (jnp.broadcast_to(jnp.asarray(c), value.shape) for c in constraint.closed)
    return jnp.where(lower_closed, lower, lower + _NUDGE), jnp.where(upper_closed, upper, upper - _NUDGE)


def _to_box(node: Param | Array) -> Array:
    """Returns the declared value of `node` in its box space."""
    value = jnp.asarray(node.value if is_param(node) else node)
    constraint = _box_constraint(node)
    if constraint is None:
        return value
    return jnp.broadcast_to(constraint.base_bijector.inverse(value), value.shape).astype(value.dtype)


def _from_box(node: Param | Array, box: Array) -> Array:
    """Returns the box-space value `box` of `node` in declared space.

    The result is kept inside the bounds, which rounding at a closed edge could
    otherwise step past. The comparison leaves the gradient at the edge intact.
    """
    constraint = _box_constraint(node)
    if constraint is None:
        return box
    declared = constraint.base_bijector.forward(box)
    lower, upper = constraint.bounds
    return jnp.where(declared < lower, lower, jnp.where(declared > upper, upper, declared))


def _node_from_box(node: Param | Array, box: Array) -> Param | Array:
    """Returns `node` holding the box-space value `box`, for evaluating the objective.

    A parameter stays a parameter, with its scale, name and validity, so a derived
    model's builder sees what it would otherwise. It holds its declared value directly,
    not through its raw value, which is infinite on a closed bound, so the value and
    its gradient are finite there.
    """
    declared = _from_box(node, box)
    if not is_param(node):
        return declared
    return dataclasses.replace(node, variable=prx.Real(declared))


def _nudged(param: Param) -> Array | None:
    """Returns the declared value of `param` with each element on a closed bound moved
    1e-6 inward in box space, or None if no element is on one.

    Box space is the constraint's Parax base space: the unit box between two finite
    bounds, declared space otherwise. An element also counts as on a bound when it is
    no further from it, in box space, than twice the bound's image through the
    raw-to-declared bijector. A Parax prior whitening clips on the way out of raw
    space, so a prior's value set on its bound reads back slightly inside it, with a
    raw value too far out to move.
    """
    constraint = prx.unwrap(param.constraint)
    if constraint is None:
        return None
    value = param.value
    to_declared = constraint.base_bijector

    def _like_value(x):
        return jnp.broadcast_to(jnp.asarray(x), value.shape)

    box = _like_value(to_declared.inverse(value))
    lower, upper = (_like_value(b).astype(value.dtype) for b in constraint.base_bounds)
    lower_closed, upper_closed = (_like_value(c) for c in constraint.closed)
    ends = [
        _like_value(to_declared.inverse(constraint.bijector.forward(jnp.full(value.shape, x, value.dtype))))
        for x in (-jnp.inf, jnp.inf)
    ]
    unknown = jnp.isnan(ends[0]) | jnp.isnan(ends[1])
    lower_image = jnp.where(unknown, lower, jnp.minimum(*ends))
    upper_image = jnp.where(unknown, upper, jnp.maximum(*ends))

    at_lower = lower_closed & jnp.isfinite(lower) & (box - lower <= 2 * (lower_image - lower))
    at_upper = upper_closed & jnp.isfinite(upper) & ~at_lower & (upper - box <= 2 * (upper - upper_image))
    if not bool(jnp.any(at_lower | at_upper)):
        return None

    nudged = to_declared.forward(jnp.where(at_lower, lower + _NUDGE, upper - _NUDGE))
    return jnp.where(at_lower | at_upper, nudged, value).astype(value.dtype)


class SolverView:
    """
    The free parameters of a model, as the dict of values a solver moves through.

    The starting values :attr:`y0` are raw, and :meth:`updated` and :meth:`objective`
    take a ``space`` so a solver working in declared space, such as a hypercube
    sampler, or in box space, such as a minimiser that honours bounds, can use the
    same view. :meth:`box` gives the start and edges in box space.

    A parameter starting on a closed bound has an infinite raw value, so :attr:`y0`
    starts it 1e-6 inward in box space instead (ADR-0007). The model keeps
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
        # The free parameters' paths and nodes, by name, for box space.
        self._free = tree_param_paths(model, free_only=True)

    def box(self) -> tuple[dict, tuple[dict, dict]]:
        """Returns the start and the box a bounded minimiser searches, in box space.

        A start on a closed bound stays on it. A start outside the box, which can only
        be within 1e-6 of an open bound, is moved onto its edge.

        Returns
        -------
        tuple
            ``(y0, (lower, upper))``: the starting box values of the free parameters
            and the lower and upper edges of the box, each a dict by name.
        """
        lower, upper, y0 = {}, {}, {}
        for name, (_, node) in self._free.items():
            lower[name], upper[name] = _box_edges(node)
            y0[name] = jnp.clip(_to_box(node), lower[name], upper[name])
        return y0, (lower, upper)

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

        `space` is ``'box'`` or any space :func:`pmrf.update` takes. Batched values give
        a batched model; fixed parameters stay unbatched.
        """
        if space == 'box':
            declared = {name: _from_box(self._free[name][1], v) for name, v in values.items()}
            return update(self.model, declared, space='declared')
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
        """Returns `fn`, which takes the unwrapped model, as a function of values in `space`.

        In box space each free parameter holds its declared value directly, rather than
        through its raw value, which is infinite on a closed bound. So the objective and
        its gradient are finite there.
        """
        if space == 'box':
            names = list(self._free)
            getter = Pathgetter(*(path for path, _ in self._free.values()))

            def box_value_fn(values: dict, args: Any) -> Scalar:
                nodes = [_node_from_box(self._free[name][1], values[name]) for name in names]
                model = eqx.tree_at(getter, self.model, nodes[0] if len(nodes) == 1 else tuple(nodes))
                return fn(unwrap(model), args)
            return box_value_fn

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
