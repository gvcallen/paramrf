"""
Adapters to hold discrete and continuous surrogate models.
"""

from typing import Callable

import jax.numpy as jnp

from pmrf.core import Frequency
from pmrf.core import Parameter
from pmrf.models.adapters.abstract import AbstractSingleProperty, AbstractSingleDiscreteProperty
    
class ContinuousSurrogate(AbstractSingleProperty, transparent=True):
    """
    A model that predicts its output at an arbitrary frequency using an arbitrary callable.
    
    This is very useful for embedding machine learning architectures. For example, `callable` can be any Equinox module.
    """
    # The input parameters of length P
    theta: Parameter | list[Parameter] = None
    
    # The underlying model. Must accept an array of shape (P,) and a frequency object, and return an array of shape nfreq x nports x nports.
    func: Callable[[jnp.ndarray, Frequency], jnp.ndarray] = None
    
    def output(self, freq: Frequency) -> jnp.ndarray:
        # Hack reshape for now
        return self.func(self.flat_param_values(include_fixed=True), freq).reshape(-1, 1, 1)
    

class DiscreteSurrogate(AbstractSingleDiscreteProperty, transparent=True):
    """
    A model that predicts its output at a discrete set of frequency values using an arbitrary callable.
    
    This is very useful for embedding machine learning architectures. For example, `callable` can be any Equinox module.
    """
    # The input parameters of length P
    theta: Parameter | list[Parameter] = None
    
    # The underlying model. Must accept an array of shape (P,) and return an array of shape self.frequency.npoints x nports x nports.
    func: Callable[[jnp.ndarray], jnp.ndarray] = None
    
    def output_discrete(self) -> jnp.ndarray:
        # Hack reshape for now
        return self.func(self.flat_param_values(include_fixed=True)).reshape(-1, 1, 1)