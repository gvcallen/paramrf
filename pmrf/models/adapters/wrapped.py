"""Adapter for exposing the RF interface of a wrapped module."""

import parax as prx

from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.models.base import Model
from pmrf.rf import MNAStamp
from pmrf.types import ArrayLike


class Wrapped(Model):
    """Adapt a module that unwraps to a model to the RF model interface."""

    module: Module

    def _model(self) -> Model:
        model = prx.unwrap(self.module)
        if not isinstance(model, Model):
            raise TypeError(
                "Wrapped.module must unwrap to a pmrf.Model; "
                f"got {type(model).__name__}."
            )
        return model

    @property
    def number_of_ports(self) -> int:
        return self._model().number_of_ports

    @property
    def primary_domain(self) -> str:
        return self._model().primary_domain

    def primary_matrix(self, frequency: Frequency, **kwargs):
        return self._model().primary_matrix(frequency, **kwargs)

    def s(self, frequency: Frequency, z0: ArrayLike = 50.0):
        return self._model().s(frequency, z0=z0)

    def a(self, frequency: Frequency):
        return self._model().a(frequency)

    def y(self, frequency: Frequency):
        return self._model().y(frequency)

    def z(self, frequency: Frequency):
        return self._model().z(frequency)

    def mna(self, frequency: Frequency) -> MNAStamp:
        return self._model().mna(frequency)

    def expand(self):
        return self._model().expand()
