from abc import abstractmethod

import numpy as np
import jax.numpy as jnp
from numpyro.distributions import Distribution

from pmrf.distributions.trainable import TrainableDistribution, TrainableDistributionT
    
class SampledDistribution(Distribution):
    @property
    def num_params(self) -> int:
        return len(self.param_names())
    
    @abstractmethod
    def param_names(self) -> list[str]:
        """
        Retrieves parameter names associated with the samples of this distribution.

        Returns
        -------
        list[str]
            The list of parameter names
        """
        raise NotImplementedError
    
    @abstractmethod
    def samples(self, weighted=False) -> jnp.ndarray:
        """
        Retrieve samples drawn from the distribution.

        Parameters
        ----------
        weighted : bool, optional, default=False
            If True, returns weighted (non-resampled) samples.

        Returns
        -------
        jnp.ndarray
            The array of samples.
        """
        raise NotImplementedError

    @abstractmethod
    def weights(self) -> jnp.ndarray:
        """
        Retrieve weights assocaited with the samples, if any.

        Returns
        -------
        jnp.ndarray
            The array of weights.
        """
        raise NotImplementedError
    
    def sample(self, key, sample_shape):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def icdf(self, u):
        raise NotImplementedError