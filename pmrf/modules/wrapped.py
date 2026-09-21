"""Modules that apply parameter-aware transformations to other modules."""

from typing import Any, Callable

import parax as prx

from pmrf.modules.base import Module


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
