"""Base models that delegate their RF interface to another model."""

from abc import ABC, abstractmethod

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.rf import MNAStamp
from pmrf.types import ArrayLike


class AbstractBuilder(Model, ABC):
    """Abstract base for an RF model implemented by building another model.

    Subclasses implement :meth:`build` and may then be used anywhere a
    :class:`pmrf.Model` is accepted. The returned model supplies the complete RF
    interface of the builder, including its port count and expanded topology.

    ``build()`` is a pure, lazy realization hook rather than a one-time
    initializer. It may be called during RF evaluation, JAX tracing, port
    discovery, topology expansion, or explicit introspection. Implementations
    must not perform I/O or mutation, and ParamRF does not cache the returned
    model. Its type, PyTree topology, and port count must remain stable for a
    given set of static fields; topology must not depend on dynamic fitted
    parameter values.
    """

    # Model uses this marker to distinguish the supported builder contract from
    # deprecated classes that override Model.build() directly.
    _pmrf_explicit_builder = True

    @abstractmethod
    def build(self) -> Model:
        """Return the model that supplies this object's RF behaviour."""
        raise NotImplementedError

    def _model(self) -> Model:
        model = self.build()
        if not isinstance(model, Model):
            raise TypeError(
                "AbstractBuilder.build() must return a pmrf.Model; "
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
