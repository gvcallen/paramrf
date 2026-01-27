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
    def from_sampled_distribution(cls, sampled_distribution: SampledDistribution, weighted=False, drift_sigma=0.0, boost_method=None, boost_samples=10000, key=None, **train_kwargs) -> 'TrainableDistribution':
        """
        Train this distribution from a sampled distribution.

        Parameters
        ----------
        train_distribution : TrainableDistributionT or None, optional
            The distribution class to train. If None, defaults to `MargarineMAFDistribution`.
        weighted : bool, optional, default=False
            If False, uses weights for training; otherwise uses equal weights.
        drift_sigma : float, optional, default=0.0
            Standard deviation for drift augmentation to broaden the posterior support.
        boost_method : str or None, optional
            Method to boost sample count ('kde' or None).
        boost_samples : int, optional, default=10000
            Number of samples to generate if boosting is enabled.
        **train_kwargs
            Additional keyword arguments passed to the distribution's training method.
        """
        logging.info(f'Training distribution on {sampled_distribution.param_names()}')
        
        training_data: jnp.ndarray = sampled_distribution.samples(weighted=weighted)[:,0:sampled_distribution.num_params]

        if drift_sigma != 0.0:
            if boost_method == 'kde':
                from margarine.estimators.kde import KDE
                kde = KDE(training_data)
                kde.train()
                training_data = kde.sample(boost_samples)
            elif boost_method != None:
                raise Exception('Unknown posterior training data boost method')
                
            scale = np.abs(np.mean(training_data, axis=0)) * drift_sigma
            training_data += np.random.normal(loc=0.0, scale=scale, size=training_data.shape)

        if not weighted:
            dist = cls.from_samples(training_data, key=key, **train_kwargs)
        else:
            weights = sampled_distribution.weights()
            dist = cls.from_weighted_samples(training_data, weights, key=key, **train_kwargs)
        
        return dist
    
    
TrainableDistributionT = TypeVar("TrainableDistributionT", bound=TrainableDistribution)