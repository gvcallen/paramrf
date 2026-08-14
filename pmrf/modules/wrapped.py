"""Modules that apply parameter-aware transformations to other modules."""

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import parax as prx
from distreqx.distributions import AbstractDistribution
from parax.constraints import AbstractConstraint

from pmrf.modules.base import Module
from pmrf.parameters import Param, tree_named_params


def _make_probabilistic_node(distribution, value, *, constraint, static):
    """A `parax.Probabilize` node, `parax.Combine`-wrapped with `static` if given."""
    from pmrf.distributions import Transformed
    from pmrf.parameters import is_param

    if is_param(value):
        to_physical = value.constrained_to_physical_bijector
        if to_physical is not None:
            distribution = Transformed(prx.as_unwrapped(distribution), to_physical)

    node = prx.Probabilize(distribution, value, constraint=constraint)
    if static is not None:
        node = prx.Combine(node, static)
    return node


class Tied(Module, prx.AbstractUnwrappable):
    """A module that ties a target field to a transformed source field."""

    tie: prx.Tie

    def __init__(
        self,
        module: Module | None = None,
        target: Callable[[Any], Any] | None = None,
        source: Callable[[Any], Any] | None = None,
        tie_fn: Callable[[Any], Any] = lambda x: x,
        *,
        model: Module | None = None,
    ):
        if module is None:
            module = model
        elif model is not None:
            raise TypeError("Pass only one of `module` or deprecated `model`.")
        if module is None:
            raise TypeError("Missing required argument: `module`.")
        if target is None or source is None:
            raise TypeError("`target` and `source` are required.")
        base_tree = module.tie if isinstance(module, Tied) else module
        self.tie = prx.Tie(base_tree, target, source, tie_fn)

    def unwrap(self) -> Module:
        return self.tie

    def named_params(
        self,
        full_params: bool = False,
        free_only: bool = False,
        namespace_separator: str = "_",
    ) -> dict[str, float | jnp.ndarray | Param]:
        """Return names relative to the wrapped module rather than the wrapper."""
        module = self.tie.tree if free_only else prx.unwrap(self)
        return tree_named_params(
            module,
            full_params=full_params,
            free_only=free_only,
            namespace_separator=namespace_separator,
        )

    @property
    def module(self) -> Module:
        """The underlying module with tied targets removed."""
        return self.tie.tree


class Probabilistic(Module, prx.AbstractUnwrappable):
    """A module with a probability distribution attached to a subtree."""

    probabilistic: prx.Probabilize | prx.Combine | Module

    def __init__(
        self,
        module: Module | None = None,
        distribution: AbstractDistribution | None = None,
        target: Callable[[Any], Any] = lambda m: m,
        constraint: AbstractConstraint | None = None,
        static: Any = None,
        *,
        model: Module | None = None,
    ):
        if module is None:
            module = model
        elif model is not None:
            raise TypeError("Pass only one of `module` or deprecated `model`.")
        if module is None:
            raise TypeError("Missing required argument: `module`.")
        if distribution is None:
            raise TypeError("Missing required argument: `distribution`.")
        target_vals = target(module)
        if isinstance(target_vals, tuple):
            if not isinstance(distribution, tuple) or len(distribution) != len(target_vals):
                raise ValueError(
                    "If 'target' returns a tuple, 'distribution' must be a tuple "
                    "of the exact same length."
                )
            if constraint is None:
                constraint_tup = (None,) * len(target_vals)
            elif not isinstance(constraint, tuple) or len(constraint) != len(target_vals):
                raise ValueError(
                    "If 'target' returns a tuple, 'constraint' must be None or a "
                    "tuple of the exact same length."
                )
            else:
                constraint_tup = constraint

            if static is None:
                static_tup = (None,) * len(target_vals)
            elif not isinstance(static, tuple) or len(static) != len(target_vals):
                raise ValueError(
                    "If 'target' returns a tuple, 'static' must be None or a tuple "
                    "of the exact same length."
                )
            else:
                static_tup = static

            prob_nodes = tuple(
                _make_probabilistic_node(dist, val, constraint=cons, static=stat)
                for dist, val, cons, stat in zip(
                    distribution, target_vals, constraint_tup, static_tup
                )
            )
            self.probabilistic = eqx.tree_at(target, module, prob_nodes)
        else:
            if isinstance(distribution, tuple):
                raise ValueError(
                    "Provided a tuple of distributions, but 'target' returned a single node."
                )
            if isinstance(constraint, tuple):
                raise ValueError(
                    "Provided a tuple of constraints, but 'target' returned a single node."
                )
            self.probabilistic = eqx.tree_at(
                target,
                module,
                _make_probabilistic_node(
                    distribution, target_vals, constraint=constraint, static=static
                ),
            )

    def unwrap(self) -> Module:
        return self.probabilistic

    @property
    def module(self) -> Module:
        """The probabilistic module tree."""
        return self.probabilistic
