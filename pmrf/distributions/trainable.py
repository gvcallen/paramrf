from typing import TypeVar

from abc import abstractmethod
import jax.numpy as jnp

from numpyro.distributions import Distribution

class TrainableDistribution(Distribution):
    @classmethod
    @abstractmethod
    def from_samples(cls, samples: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_weighted_samples(cls, samples: jnp.ndarray, weights: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError
    
    
    
TrainableDistributionT = TypeVar("TrainableDistributionT", bound=TrainableDistribution)