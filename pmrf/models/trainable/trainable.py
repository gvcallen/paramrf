from typing import Self
from abc import ABC, abstractmethod
import jax.numpy as jnp

import skrf as rf
import pmrf as prf
from pmrf.models import Model
from pmrf._util import remove_constant_params

from typing import Self
from abc import ABC, abstractmethod
import jax.numpy as jnp
import pmrf as prf
from pmrf.models import Model  # Your base generic model class
from pmrf._util import remove_constant_params

class Trainable(Model, ABC):
    """
    A model where some feature, such as S-parameters, is trained based samples of that parameter.
    """

    @classmethod
    def from_networks(cls, networks: prf.NetworkCollection, nominal_network: rf.Network | None = None, feature='s', **kwargs) -> Self:
        networks = networks.interpolate()
        samples = jnp.stack([jnp.array(getattr(ntwk, feature)) for ntwk in networks], dtype=getattr(networks[0], feature).dtype)
        nominal_sample = getattr(nominal_network, feature) if nominal_network is not None else None
        freq = prf.Frequency.from_skrf(networks[0])
        return cls.from_samples(samples, freq, nominal_sample=nominal_sample, feature=feature, **kwargs)

    @classmethod
    @abstractmethod
    def from_samples(cls, samples: jnp.ndarray, frequency: prf.Frequency, nominal_sample: jnp.ndarray | None = None, feature='s', **kwargs) -> Self:
        raise NotImplementedError

class ParametricTrainable(Model, ABC):
    @classmethod
    def from_parametric_networks(cls, networks: prf.NetworkCollection, nominal_network: rf.Network | None = None, feature='s', **kwargs) -> Self:
        networks = networks.interpolate()
        samples = jnp.stack([jnp.array(ntwk.s) for ntwk in networks])
        nominal_sample = getattr(nominal_network, feature) if nominal_network is not None else None
        frequency = prf.Frequency.from_skrf(networks[0])
        
        params_list: list[dict[str, float]] = remove_constant_params([n.params for n in networks])
        nominal_params = {k: v for k, v in nominal_network.params.items() if k in params_list[0].keys()} if nominal_network is not None else None
        params = jnp.stack([jnp.array(ntwk_params.values()) for ntwk_params in params_list], axis=-1)

        return cls.from_parametric_samples(params, samples, frequency, nominal_sample=nominal_sample, nominal_params=nominal_params, feature=feature, **kwargs)

    @classmethod
    @abstractmethod
    def from_parametric_samples(cls, params: jnp.ndarray, samples: jnp.ndarray, frequency: prf.Frequency, nominal_sample: jnp.ndarray | None = None, nominal_params: jnp.ndarray | None = None, feature='s', **kwargs) -> Self:
        raise NotImplementedError