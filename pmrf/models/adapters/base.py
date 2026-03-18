"""
Abstract base adapter models.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.core import Frequency, Model
from pmrf.core.field import field
from pmrf.utils import is_overridden

class Discrete(Model, ABC):
    """
    A model whose properties are defined on a tabulated frequency grid.
    
    To use, set self.frequency and override one or more of the `xxx_discrete` methods.
    The base Model conversions (s2a, s2z, etc.) will be applied automatically
    to the interpolated values.
    """
    frequency: Frequency = None

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
        if is_overridden(type(self), Discrete, 's_discrete'):
            return self._interp(self.s_discrete(), freq)
        return super().s(freq)

    @eqx.filter_jit
    def a(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), Discrete, 'a_discrete'):
            return self._interp(self.a_discrete(), freq)
        return super().a(freq)

    @eqx.filter_jit
    def y(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), Discrete, 'y_discrete'):
            return self._interp(self.y_discrete(), freq)
        return super().y(freq)

    @eqx.filter_jit
    def z(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), Discrete, 'z_discrete'):
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


class SingleProperty(Model, ABC):
    """
    A model that acts as a wrapper around a single known property type 
    (e.g., a data file that only contains S-parameters).
    """
    # property: str = field(static=True, default='s') # 's', 'a', 'y', or 'z'
    property: str = 's'
    
    @abstractmethod
    def output(self, freq: Frequency) -> jnp.ndarray:
        """The primary computation for the chosen property."""
        raise NotImplementedError
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.property == 's' else super().s(freq)

    def a(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.property == 'a' else super().a(freq)

    def y(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.property == 'y' else super().y(freq)

    def z(self, freq: Frequency) -> jnp.ndarray:
        return self.output(freq) if self.property == 'z' else super().z(freq)


class SingleDiscreteProperty(SingleProperty, Discrete, ABC):
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
        if self.property == 's': return self.output_discrete()
        raise NotImplementedError

    def a_discrete(self) -> jnp.ndarray:
        if self.property == 'a': return self.output_discrete()
        raise NotImplementedError

    def y_discrete(self) -> jnp.ndarray:
        if self.property == 'y': return self.output_discrete()
        raise NotImplementedError

    def z_discrete(self) -> jnp.ndarray:
        if self.property == 'z': return self.output_discrete()
        raise NotImplementedError    
