"""
Base adapter models used as abstract bases for concrete adapters.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils.type import is_overridden
from pmrf.utils import field, freeze

class AbstractDiscrete(Model, ABC):
    """
    (experimental) A model whose properties are defined on a discrete (tabulated) frequency grid.
    
    To use, set `self.frequency` and override one or more of the `xxx_discrete` methods.
    The base Model conversions (s2a, s2z, etc.) will be applied automatically
    to the interpolated values.

    Parameters
    ----------
    frequency : Frequency
        The constant frequency over which the discrete model is defined.
    """
    #: The constant frequency.
    frequency: Frequency = field(converter=freeze)

    # Tabulated data entry points
    def s_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def a_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def y_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def z_discrete(self) -> jnp.ndarray: raise NotImplementedError

    @eqx.filter_jit
    def s(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 's_discrete'):
            return self._interp(self.s_discrete(), freq)
        return super().s(freq)

    @eqx.filter_jit
    def a(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 'a_discrete'):
            return self._interp(self.a_discrete(), freq)
        return super().a(freq)

    @eqx.filter_jit
    def y(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 'y_discrete'):
            return self._interp(self.y_discrete(), freq)
        return super().y(freq)

    @eqx.filter_jit
    def z(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 'z_discrete'):
            return self._interp(self.z_discrete(), freq)
        return super().z(freq)

    def _interp(self, x: jnp.ndarray, freq: Frequency) -> jnp.ndarray:
        """
        Vectorized interpolation across port matrices without moveaxis.
        """
        f_new = freq.f_scaled
        f_old = self.frequency.f_scaled
        
        def interp_trace(trace):
            return jnp.interp(f_new, f_old, trace)

        vmap_cols = jax.vmap(interp_trace, in_axes=1, out_axes=1)
        vmap_matrix = jax.vmap(vmap_cols, in_axes=1, out_axes=1)
        
        return vmap_matrix(x)


class AbstractSingleProperty(Model, ABC):
    """
    A model that acts as a wrapper around a single known property type
    (e.g., a data file that only contains S-parameters)
    which caters for dynamic injection of the kind of property matrix.
    """
    kind: str = eqx.field(default='s', static=True, kw_only=True)

    @property
    def primary_property(self) -> str:
        return self.kind    
    
    @abstractmethod
    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError


class AbstractSingleDiscreteProperty(AbstractSingleProperty, AbstractDiscrete, ABC):
    """
    (experimental) A model that provides a single dynamic property type from a discrete grid.
    
    To override this class, implement `discrete_matrix` which must return a matrix property
    of type `self.kind`.
    """    
    @abstractmethod
    def discrete_matrix(self) -> jnp.ndarray:
        """The primary discrete data."""
        raise NotImplementedError
    
    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        return self._interp(self.discrete_matrix(), freq)