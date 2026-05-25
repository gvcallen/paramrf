"""
Adapters that wrap callables representing external models.
"""

from typing import Callable

import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.models.adapters.base import AbstractSingleDomain, AbstractSingleDiscreteDomain
from pmrf.utils import unfreeze, field, unwrap_self
from pmrf.types import Param
from pmrf.parameters import param
    
class ContinuousCallable(AbstractSingleDomain):
    """
    A model that predicts its output at an arbitrary frequency using an arbitrary callable.
    
    This class can be used to wrap external machine learning architectures (Equinox/other).
    
    Parameters
    ----------
    fn : Callable[[jnp.ndarray], jnp.ndarray] | Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        The underlying callable model which predicts the response as a function of scaled frequency.
        May either be a function or a callable PyTree (e.g. :class:`equinox.Module`) which will be frozen.
        Must accept an array of shape `(nfreq,)` or `(nfreq, nparams)` depending on if `theta` is None,
        and return an array of shape `(nfreq, nports, nports)`.
    theta : Param
        Parameters to pass to `fn` of length `nparams`.
        Can be None for models that contain their own :class:`parax.Parameter` objects.
        All parameters, including fixed parameters, are passed.
    """
    #: The underlying callable model
    fn: Callable[[jnp.ndarray], jnp.ndarray] | Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = field(converter=unfreeze)
    
    #: Parameters to pass to fn
    theta: Param = param()
    
    @eqx.filter_jit
    @unwrap_self
    def matrix(self, freq: Frequency) -> jnp.ndarray:
        if self.theta is not None:
            flat_theta = jnp.array(self.theta)
            mat = self.fn(freq.f_scaled, flat_theta)
        else:
            mat = self.fn(freq.f_scaled)
            
        return jnp.asarray(mat, dtype=jnp.complex128)
    

class DiscreteCallable(AbstractSingleDiscreteDomain):
    """
    A model that predicts its output at a discrete set of frequencies already known to the model using an arbitrary callable.
    
    This class can be used to wrap external machine learning architectures (Equinox/Parax/other).

    Parameters
    ----------
    fn : Callable[[], jnp.ndarray] | Callable[[jnp.ndarray], jnp.ndarray]
        The underlying callable model which predicts the response.
        May either be a function or a callable PyTree (e.g. :class:`equinox.Module`) which will be frozen.
        Must either accept no parameters or an array of shape `(nparams,)` depending on if `theta` is None,
        and return an array of shape `(nfreq, nports, nports)`.
    theta : Param
        Parameters to pass to `fn` of length `nparams`.
        Can be None for models that contain their own :class:`parax.Parameter` objects.
        All parameters, including fixed parameters, are passed.
    """
    #: The underlying callable model
    fn: Callable[[], jnp.ndarray] | Callable[[jnp.ndarray], jnp.ndarray] = field(converter=unfreeze)
    
    #: Parameters to pass to fn
    theta: Param = param()
    
    @eqx.filter_jit
    @unwrap_self
    def discrete_matrix(self) -> jnp.ndarray:
        if self.theta is not None:
            flat_theta = jnp.array(self.theta)
            mat = self.fn(flat_theta)
        else:
            mat = self.fn()
            
        return jnp.asarray(mat, dtype=jnp.complex128)