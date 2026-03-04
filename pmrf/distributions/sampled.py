"""
A distribution represented by a set of sampled.
"""

from abc import abstractmethod

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution

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
    
    def sample(self, key, sample_shape=()):
        # Ensure sample_shape is a tuple
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
            
        available_samples = self.samples()
        num_available = available_samples.shape[0]
        
        # Strategy: Sample uniformly with replacement using random integers.
        # This allows us to draw any number of samples (larger or smaller than num_available)
        # from the fixed underlying empirical distribution without running out of samples.
        indices = jax.random.randint(
            key, 
            shape=sample_shape, 
            minval=0, 
            maxval=num_available
        )
        
        return available_samples[indices]

    def log_prob(self, value):
        raise NotImplementedError

    def icdf(self, u):
        raise NotImplementedError