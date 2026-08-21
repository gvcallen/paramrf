"""Adapter for exposing the RF interface of a wrapped module."""

import parax as prx

from pmrf.modules.base import Module
from pmrf.models.base import Model
from pmrf.models.adapters.delegated import AbstractBuilder


class Wrapped(AbstractBuilder):
    """Adapt a module that unwraps to a model to the RF model interface."""

    wrapped: Module

    def build(self) -> Model:
        model = prx.unwrap(self.wrapped)
        if not isinstance(model, Model):
            raise TypeError(
                "Wrapped.wrapped must unwrap to a pmrf.Model; "
                f"got {type(model).__name__}."
            )
        return model
