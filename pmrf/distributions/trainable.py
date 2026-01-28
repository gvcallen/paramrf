import logging
from typing import TypeVar
from abc import abstractmethod
import jax.numpy as jnp

from numpyro.distributions import Distribution
import numpy as np

from pmrf.distributions.sampled import SampledDistribution

class TrainableDistribution(Distribution):
    @classmethod
    @abstractmethod
    def from_samples(cls, samples: jnp.ndarray, weights: jnp.ndarray | None = None, key=None, init_kwargs=None, **train_kwargs):
        raise NotImplementedError

    @classmethod
    def from_sampled_distribution(cls, sampled_distribution: SampledDistribution, weighted=False, key=None, init_kwargs=None, **train_kwargs) -> 'TrainableDistribution':
        """
        Train this distribution from a sampled distribution.

        Parameters
        ----------
        train_distribution : TrainableDistributionT or None, optional
            The distribution class to train. If None, defaults to `MargarineMAFDistribution`.
        weighted : bool, optional, default=False
            If False, uses weights for training; otherwise uses equal weights.
        **train_kwargs
            Additional keyword arguments passed to the distribution's training method.
        """
        training_data: jnp.ndarray = sampled_distribution.samples(weighted=weighted)[:,0:sampled_distribution.num_params]

        # Formula to broaden the flow
        # scale = np.abs(np.mean(training_data, axis=0)) * drift_sigma
        # training_data += np.random.normal(loc=0.0, scale=scale, size=training_data.shape)
        
        logging.info(f'Training distribution on {sampled_distribution.param_names()}')

        if not weighted:
            dist = cls.from_samples(training_data, key=key, init_kwargs=init_kwargs, **train_kwargs)
        else:
            dist = cls.from_samples(training_data, weights=sampled_distribution.weights(), key=key, init_kwargs=init_kwargs, **train_kwargs)
        
        return dist
    
    
TrainableDistributionT = TypeVar("TrainableDistributionT", bound=TrainableDistribution)