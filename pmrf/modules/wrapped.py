"""Modules that apply parameter-aware transformations to other modules."""

from typing import Any, Callable

import parax as prx

from pmrf.distributions import AbstractDistribution
from pmrf.modules.base import Module
from pmrf.parameters import Space
from pmrf.utils import field, freeze


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

    @property
    def module(self) -> Module:
        """The underlying module with tied targets removed."""
        return self.tie.tree


class Probabilistic(Module, prx.AbstractUnwrappable):
    """
    A joint prior over named parameters of a tree, which it wraps unchanged.

    Built by :func:`pmrf.prior` (ADR-0005). The parameters keep their names and stay
    parameters: names are resolved straight through this wrapper, as through
    :class:`Tied`. The joint prior replaces the parameters' own priors in
    :func:`pmrf.log_prior`. :func:`pmrf.resolve` keeps it, and :func:`pmrf.unwrap`
    drops it, as it drops every prior.
    """

    #: The wrapped tree, unchanged.
    module: Any

    #: The joint distribution, over the vector of the parameters' values in `space`.
    #: It is held frozen, so its arrays are not parameters.
    distribution: AbstractDistribution = field(converter=freeze)

    #: The parameter names, relative to `module`, in the order of the distribution's vector.
    names: tuple[str, ...] = field(static=True)

    #: The space the distribution is over: ``'raw'``, ``'declared'`` or ``'physical'``.
    #: ``'raw'`` is the parameters' raw space as it was when the prior was attached.
    space: Space = field(static=True)

    def unwrap(self) -> Any:
        return self.module
