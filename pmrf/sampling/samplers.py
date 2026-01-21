import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from pmrf.sampling._base import BaseSampler

class LatinHypercubeSampler(BaseSampler):
    """
    Sampler using Latin Hypercube Sampling (LHS).

    LHS is a stratified sampling method that generates sample points that are
    more evenly distributed across the hypercube than standard random sampling.
    This implementation uses `scipy.stats.qmc.LatinHypercube`.
    """
    def _generate_hypercube_samples(self, N, D) -> jnp.ndarray:
        """
        Generate samples using Latin Hypercube Sampling.

        Parameters
        ----------
        N : int
            Number of samples.
        D : int
            Dimensionality (number of parameters).

        Returns
        -------
        jnp.ndarray
            Samples in the unit hypercube `[0, 1)^D`.
        """
        return jnp.array(qmc.LatinHypercube(D).random(N))

class UniformSampler(BaseSampler):
    """
    Sampler using standard Uniform Random Sampling (Monte Carlo).

    Samples are drawn independently from a uniform distribution over the
    unit hypercube.
    """
    def _generate_hypercube_samples(self, N, D) -> jnp.ndarray:
        """
        Generate samples using uniform random sampling.

        Parameters
        ----------
        N : int
            Number of samples.
        D : int
            Dimensionality (number of parameters).

        Returns
        -------
        jnp.ndarray
            Samples in the unit hypercube `[0, 1)^D`.
        """
        return jnp.array(np.random.uniform(0.0, 1.0, size=(N, D)))