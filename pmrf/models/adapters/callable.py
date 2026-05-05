"""
Adapters that wrap callables representing external models.
"""

from typing import Callable, Sequence

import jax.numpy as jnp

from pmrf.core import Frequency
from pmrf.parameters import Parameter
from pmrf.models.adapters.base import SingleProperty, SingleDiscreteProperty
from pmrf.field import frozen
import parax as prx
    
class ContinuousCallable(SingleProperty):
    """
    A model that predicts its output at an arbitrary frequency using an arbitrary callable.
    
    This class can be used to wrap external machine learning architectures (Equinox/Parax/other).
    """
    #: Parameters to pass to `fn` of length `nparams`.
    #: Can be None for models that contain their own :class:`parax.Parameter` objects.
    #: All parameters, including fixed parameters, are passed.
    #: If a list is provided, the parameters are first stacked.
    theta: Parameter | list[Parameter]
    
    #: The underlying callable model which predicts the response as a function of scaled frequency.
    #: May either be a function or a callable PyTree (e.g. :class:`parax.Module`) with optional internal parameters.
    #: Must accept an array of shape `(nfreq,)` or `(nparams, nfreq,)` depending on if `theta` is None,
    #: and return an array of shape `(nfreq, nports, nports)`.
    fn: Callable[[jnp.ndarray], jnp.ndarray] | Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = frozen()
    
    def output(self, freq: Frequency) -> jnp.ndarray:
        if self.theta is not None:
            if isinstance(self.theta, Sequence):
                flat_theta = jnp.stack([jnp.array(ti) for ti in self.theta])
            else:
                flat_theta = jnp.array(self.theta)

            # return self.fn(self.flat_param_values(include_fixed=True)).reshape(-1, 1, 1)
            return self.fn(flat_theta, freq.f_scaled)
        else:
            return self.fn(freq.f_scaled)
    

class DiscreteCallable(SingleDiscreteProperty):
    """
    A model that predicts its output at a discrete set of frequencies already known to the model using an arbitrary callable.
    
    This class can be used to wrap external machine learning architectures (Equinox/Parax/other).
    """
    #: Parameters to pass to `fn` of length `nparams`.
    #: Can be None for models that contain their own :class:`parax.Parameter` objects.
    #: All parameters, including fixed parameters, are passed.
    #: If a list is provided, the parameters are first stacked.
    theta: prx.Param | list[prx.Param]
    
    #: The underlying callable model which predicts the response.
    #: May either be a function or a callable PyTree (e.g. :class:`parax.Module`) with optional internal parameters.
    #: Must either accept no parameters or an array of shape `(nparams,)` depending on if `theta` is None,
    #: and return an array of shape `(nfreq, nports, nports)`.
    fn: Callable[[], jnp.ndarray] | Callable[[jnp.ndarray], jnp.ndarray] = frozen()
    
    def output_discrete(self) -> jnp.ndarray:
        if self.theta is not None:
            if isinstance(self.theta, Sequence):
                flat_theta = jnp.stack([jnp.array(ti) for ti in self.theta])
            else:
                flat_theta = jnp.array(self.theta)

            # return self.fn(self.flat_param_values(include_fixed=True)).reshape(-1, 1, 1)
            return self.fn(flat_theta)
        else:
            return self.fn()