from typing import Self
from abc import ABC, abstractmethod
import jax.numpy as jnp

import jax
import skrf as rf
from pmrf.models import Model
from pmrf._util import remove_constant_params

from typing import Self
from abc import ABC, abstractmethod
import jax.numpy as jnp
from pmrf.network_collection import NetworkCollection
from pmrf.frequency import Frequency
from pmrf.models import Model  # Your base generic model class
from pmrf._util import remove_constant_params, field

class BlackBox(Model, ABC):
    """
    A blackbox model is one that estimates a single property, such as 's', 'y' etc., using some underlying prediction method.
    Usually, the parameters for this prediction are trained directly from data as opposed to using some output-based fitting method.
    """
    frequency: Frequency = field(static=True)
    property: str = field(static=True)
    
    def predict(self, freq: Frequency) -> jnp.ndarray:
        # The predict function, which predicts a given parameter for a given frequency
        f_new, f_old = freq.f_scaled, self.frequency.f_scaled
        sample = self.forward()
        
        vmap_m = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)
        vmap_mn = jax.vmap(vmap_m, in_axes=(None, None, 2), out_axes=2)
        return vmap_mn(f_new, f_old, sample)
    
    def transform(self, ntwk: rf.Network) -> Self:
        # The transform function, which transform a network sample into an instance of self
        ntwk_interp = ntwk.interpolate(self.frequency.to_skrf())
        sample = getattr(ntwk_interp, self.property)
        return self.with_params(self.inverse(sample))
    
    @abstractmethod
    def forward(self) -> jnp.ndarray:
        # The forward model, which produces a sample for the current parameters
        raise NotImplementedError
        
    @abstractmethod
    def inverse(self, sample: jnp.ndarray) -> jnp.ndarray:
        # The inverse model, which produces parameters for a given sample
        raise NotImplementedError
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 'a':
            return self.predict(freq)
        else:
            return super().a(freq)

    def s(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 's':
            return self.predict(freq)
        else:
            return super().s(freq)

    def y(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 'y':
            return self.predict(freq)
        else:
            return super().y(freq)

    def z(self, freq: Frequency) -> jnp.ndarray:
        if self.property == 'z':
            return self.predict(freq)
        else:
            return super().z(freq)

class UnsupervisedBlackBox(BlackBox, ABC):
    """
    A model that can be trained in an unsupervised manner from output samples of some measured feature, such as S-parameters.
    """
    def from_networks(self, networks: NetworkCollection, property='s', **kwargs) -> Self:
        networks = networks.interpolate()
        samples = jnp.stack([jnp.array(getattr(ntwk, property)) for ntwk in networks], dtype=getattr(networks[0], property).dtype)
        frequency = Frequency.from_skrf(networks[0].frequency)
        return self.from_samples(samples, frequency, property=property, **kwargs)

    @abstractmethod
    def from_samples(cls, features: jnp.ndarray, frequency: Frequency, property='s', **kwargs) -> Self:
        raise NotImplementedError

class SupervisedBlackBox(BlackBox, ABC):
    """
    A model that can be trained in a supervised manner from parametric input-output samples of some measured feature, such as S-parameters.
    """    
    @classmethod
    def from_networks(cls, networks: NetworkCollection, property='s', **kwargs) -> Self:
        networks = networks.interpolate()
        samples = jnp.stack([jnp.array(getattr(ntwk, property)) for ntwk in networks])
        frequency = Frequency.from_skrf(networks[0].frequency)
        
        params_list: list[dict[str, float]] = remove_constant_params([n.params for n in networks])
        params = jnp.stack([jnp.array(ntwk_params.values()) for ntwk_params in params_list], axis=-1)

        return cls.from_samples(params, samples, frequency, property=property, **kwargs)

    @classmethod
    @abstractmethod
    def from_samples(cls, params: jnp.ndarray, features: jnp.ndarray, frequency: Frequency, property='s', **kwargs) -> Self:
        raise NotImplementedError