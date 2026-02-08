from typing import Self
from abc import ABC, abstractmethod
import jax.numpy as jnp

import skrf as rf
from pmrf.models import Model
from pmrf._util import remove_constant_params

from typing import Self
from abc import ABC, abstractmethod
import jax.numpy as jnp
from pmrf.network_collection import NetworkCollection
from pmrf.frequency import Frequency
from pmrf.models import Model  # Your base generic model class
from pmrf._util import remove_constant_params

class BlackBox(Model, ABC):
    """
    A blackbox model is one that estimates a single feature, such as 's', 'y' etc., using some underlying prediction method.
    Usually, the parameters for this prediction are trained directly from data as opposed to using some output-based fitting method.
    """
    feature = 's'
    
    @abstractmethod
    def predict(self, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        if self.feature == 'a':
            return self.predict(freq)
        else:
            return super().a(freq)

    def s(self, freq: Frequency) -> jnp.ndarray:
        if self.feature == 's':
            return self.predict(freq)
        else:
            return super().s(freq)

    def y(self, freq: Frequency) -> jnp.ndarray:
        if self.feature == 'y':
            return self.predict(freq)
        else:
            return super().y(freq)

    def z(self, freq: Frequency) -> jnp.ndarray:
        if self.feature == 'z':
            return self.predict(freq)
        else:
            return super().z(freq)

class UnsupervisedBlackBox(BlackBox, ABC):
    """
    A model that can be trained in an unsupervised manner from output samples of some measured feature, such as S-parameters.
    """
    @classmethod
    def from_networks(cls, networks: NetworkCollection, nominal_network: rf.Network | None = None, feature='s', **kwargs) -> Self:
        networks = networks.interpolate()
        samples = jnp.stack([jnp.array(getattr(ntwk, feature)) for ntwk in networks], dtype=getattr(networks[0], feature).dtype)
        nominal_sample = getattr(nominal_network, feature) if nominal_network is not None else None
        freq = Frequency.from_skrf(networks[0])
        return cls.from_samples(samples, freq, nominal_sample=nominal_sample, feature=feature, **kwargs)

    @classmethod
    @abstractmethod
    def from_samples(cls, samples: jnp.ndarray, frequency: Frequency, nominal_sample: jnp.ndarray | None = None, feature='s', **kwargs) -> Self:
        raise NotImplementedError

class SupervisedBlackBox(BlackBox, ABC):
    """
    A model that can be trained in a supervised manner from parametric input-output samples of some measured feature, such as S-parameters.
    """    
    @classmethod
    def from_networks(cls, networks: NetworkCollection, nominal_network: rf.Network | None = None, feature='s', **kwargs) -> Self:
        networks = networks.interpolate()
        samples = jnp.stack([jnp.array(getattr(ntwk, feature)) for ntwk in networks])
        nominal_sample = getattr(nominal_network, feature) if nominal_network is not None else None
        frequency = Frequency.from_skrf(networks[0])
        
        params_list: list[dict[str, float]] = remove_constant_params([n.params for n in networks])
        nominal_params = {k: v for k, v in nominal_network.params.items() if k in params_list[0].keys()} if nominal_network is not None else None
        params = jnp.stack([jnp.array(ntwk_params.values()) for ntwk_params in params_list], axis=-1)

        return cls.from_samples(params, samples, frequency, nominal_sample=nominal_sample, nominal_params=nominal_params, feature=feature, **kwargs)

    @classmethod
    @abstractmethod
    def from_samples(cls, params: jnp.ndarray, samples: jnp.ndarray, frequency: Frequency, nominal_sample: jnp.ndarray | None = None, nominal_params: jnp.ndarray | None = None, feature='s', **kwargs) -> Self:
        raise NotImplementedError