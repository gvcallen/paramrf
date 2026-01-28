import logging
from typing import TypeVar
from abc import abstractmethod

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution

from pmrf.distributions.sampled import SampledDistribution

class TrainableDistribution(Distribution):
    @classmethod
    @abstractmethod
    def from_samples(cls, samples: jnp.ndarray, weights: jnp.ndarray | None = None, key: jnp.ndarray | None = None, init_kwargs: dict | None = None, **train_kwargs):
        """
        Train this distribution directly from samples.

        Parameters
        ----------
        samples : jnp.ndarray
            The samples to use.
        weights: jnp.ndarray, optional
            The weights to use for training, if any. Defaults to ``None``.
        key : jnp.ndarray
            The random key for training.
        init_kwargs : dict, optional
            Initialization arguments passed to the underlying distribution for initialization.
        **train_kwargs
            Training arguments passed to the underlying distribution for training.
        """
        raise NotImplementedError

    @classmethod
    def from_sampled_distribution(cls, sampled_distribution: SampledDistribution, weighted: bool = False, key: jnp.array | None = None, init_kwargs: dict | None = None, **train_kwargs) -> 'TrainableDistribution':
        """
        Train this distribution on a sampled distribution.

        Parameters
        ----------
        sampled_distribution : SampledDistribution
            The sampled distribution to train this one on.
        weighted : bool, optional, default=False
            If False, uses weights for training; otherwise uses equal weights.
        key : jnp.ndarray
            The random key for training.
        init_kwargs : dict, optional
            Initialization arguments passed to the underlying distribution for initialization.
        **train_kwargs
            Training arguments passed to the underlying distribution for training.
        """
        training_data: jnp.ndarray = sampled_distribution.samples(weighted=weighted)[:,0:sampled_distribution.num_params]
        logging.info(f'Training distribution on {sampled_distribution.param_names()}')

        if not weighted:
            dist = cls.from_samples(training_data, key=key, init_kwargs=init_kwargs, **train_kwargs)
        else:
            dist = cls.from_samples(training_data, weights=sampled_distribution.weights(), key=key, init_kwargs=init_kwargs, **train_kwargs)
        
        return dist
    
    @classmethod
    def from_distribution(cls, distribution: Distribution, num_samples: int = 10000, key: jnp.array | None = None, init_kwargs: dict | None = None, **train_kwargs) -> 'TrainableDistribution':
        """
        Train this distribution on another distribution.

        Parameters
        ----------
        distribution : Distribution
            The distribution to train this one on.
        num_samples: int
            The number of samples to get for training data. Defaults to 10000.
        key : jnp.ndarray
            The random key for training.
        init_kwargs : dict, optional
            Initialization arguments passed to the underlying distribution for initialization.
        **train_kwargs
            Training arguments passed to the underlying distribution for training.
        """
        sample_key, train_key = jax.random.split(key)
        training_data: jnp.ndarray = distribution.sample(sample_key, sample_shape=(num_samples,))
        dist = cls.from_samples(training_data, key=train_key, init_kwargs=init_kwargs, **train_kwargs)
        
        return dist    
    
    
TrainableDistributionT = TypeVar("TrainableDistributionT", bound=TrainableDistribution)