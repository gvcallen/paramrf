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
from pmrf.jax_utils import field, freeze

class AbstractDiscrete(Model, ABC):
    """
    A model whose properties are defined on a discrete (tabulated) frequency grid.
    
    To use, set self.frequency and override one or more of the `xxx_discrete` methods.
    The base Model conversions (s2a, s2z, etc.) will be applied automatically
    to the interpolated values.
    """

    #: The constant frequency over which the discrete model is defined.
    frequency: Frequency = field(converter=freeze)

    # Tabulated data entry points
    def s_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def a_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def y_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def z_discrete(self) -> jnp.ndarray: raise NotImplementedError

    # -----------------------------------------------------------------------
    # Overriding Model dispatch to inject interpolation
    # -----------------------------------------------------------------------
    
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
        
        # 1. Base interpolator for a single trace: (F_old,) -> (F_new,)
        def interp_trace(trace):
            return jnp.interp(f_new, f_old, trace)

        # 2. Vectorize over columns (Axis 2 of input): 
        #    Input slice is (F, N) -> Output slice is (F_new, N)
        #    in_axes=1 tells vmap to iterate over the second dimension (N)
        vmap_cols = jax.vmap(interp_trace, in_axes=1, out_axes=1)

        # 3. Vectorize over rows (Axis 1 of input):
        #    Input is (F, M, N) -> Output is (F_new, M, N)
        #    in_axes=1 tells vmap to iterate over the second dimension (M)
        vmap_matrix = jax.vmap(vmap_cols, in_axes=1, out_axes=1)
        
        return vmap_matrix(x)


class AbstractSingleProperty(Model, ABC):
    """
    A model that acts as a wrapper around a single known property type 
    (e.g., a data file that only contains S-parameters).
    """
    kind: str = eqx.field(default='s', static=True, kw_only=True)

    @property
    def primary_property(self) -> str:
        # Prevents infinite recursion by explicitly telling Model the primary type
        return self.kind    
    
    @abstractmethod
    def output(self, freq: Frequency) -> jnp.ndarray:
        """The primary computation for the chosen property."""
        raise NotImplementedError
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.kind == 's' else super().s(freq)

    def a(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.kind == 'a' else super().a(freq)

    def y(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.kind == 'y' else super().y(freq)

    def z(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.kind == 'z' else super().z(freq)


class AbstractSingleDiscreteProperty(AbstractSingleProperty, AbstractDiscrete, ABC):
    """
    A model that provides a single property type from a tabulated grid.
    """    
    @abstractmethod
    def output_discrete(self) -> jnp.ndarray:
        """The primary tabulated data."""
        raise NotImplementedError
    
    # Implementation of SingleProperty.output via interpolation
    def output(self, freq: Frequency) -> jnp.ndarray:
        return self._interp(self.output_discrete(), freq)
    
    # We override the discrete methods to tell the framework which data we have.
    # If the requested discrete property doesn't match self.property, we leave
    # it as NotImplemented, forcing the Model to use high-level conversions.
    
    def s_discrete(self) -> jnp.ndarray:
        if self.kind == 's': return self.output_discrete()
        raise NotImplementedError

    def a_discrete(self) -> jnp.ndarray:
        if self.kind == 'a': return self.output_discrete()
        raise NotImplementedError

    def y_discrete(self) -> jnp.ndarray:
        if self.kind == 'y': return self.output_discrete()
        raise NotImplementedError

    def z_discrete(self) -> jnp.ndarray:
        if self.kind == 'z': return self.output_discrete()
        raise NotImplementedError    
