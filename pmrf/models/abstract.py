from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.models import Model
from pmrf._util import field

from pmrf._util import is_overridden

class Interpolated(Model, ABC):
    """
    A model whose properties (such as 's', 'y' etc.) are interpolated from a specific frequency.
    
    To use this model, set self.frequency and override any of s_discrete, a_discrete, y_discrete or z_discrete.
    """
    frequency: Frequency
    
    def s_discrete(self) -> jnp.ndarray:
        raise NotImplementedError

    def a_discrete(self) -> jnp.ndarray:
        raise NotImplementedError

    def y_discrete(self) -> jnp.ndarray:
        raise NotImplementedError

    def z_discrete(self) -> jnp.ndarray:
        raise NotImplementedError
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), Model, 'a_discrete'):        
            return self._interp(self.a_discrete())
        else:
            return super().a(freq)

    def s(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), Model, 's_discrete'):        
            return self._interp(self.s_discrete())
        else:
            return super().s(freq)

    def y(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), Model, 'y_discrete'):        
            return self._interp(self.y_discrete())
        else:
            return super().y(freq)

    def z(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), Model, 'z_discrete'):
            return self._interp(self.z_discrete())
        else:
            return super().a(freq)
    
    def _interp(self, x, freq: Frequency) -> jnp.ndarray:
        # The interpolated output function, which returns the current output interpolated onto the passed in frequency.
        # This is called by s, a, y and z depending on the current property
        f_new, f_old = freq.f_scaled, self.frequency.f_scaled
        
        vmap_m = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)
        vmap_mn = jax.vmap(vmap_m, in_axes=(None, None, 2), out_axes=2)
        return vmap_mn(f_new, f_old, x)
    
class SingleProperty(Model, ABC):
    """
    A model that can only predict a single property.
    """
    property: str = field(static=True)
    
    @abstractmethod
    def output(self, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 'a':
            return self.output(freq)
        else:
            return super().a(freq)

    def s(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 's':
            return self.output(freq)
        else:
            return super().s(freq)

    def y(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 'y':
            return self.output(freq)
        else:
            return super().y(freq)

    def z(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 'z':
            return self.output(freq)
        else:
            return super().z(freq)
        
class InterpolatedSingleProperty(Interpolated, SingleProperty, ABC):
    """
    A model that can only predict a single property that is also interpolated.
    """    
    @abstractmethod
    def output_discrete(self) -> jnp.ndarray:
        raise NotImplementedError
    
    def output(self, freq: Frequency) -> jnp.ndarray:
        return self._interp(self.output_discrete(), freq)
    
    def a_discrete(self) -> jnp.ndarray:
        if self.property == 'a':
            return self.output_discrete()
        else:
            return super().a(self.frequency)

    def s_discrete(self) -> jnp.ndarray:
        if self.property == 's':
            return self.output_discrete()
        else:
            return super().s(self.frequency)

    def y_discrete(self) -> jnp.ndarray:
        if self.property == 'y':
            return self.output_discrete()
        else:
            return super().y(self.frequency)

    def z_discrete(self) -> jnp.ndarray:
        if self.property == 'z':
            return self.output_discrete()
        else:
            return super().z(self.frequency)