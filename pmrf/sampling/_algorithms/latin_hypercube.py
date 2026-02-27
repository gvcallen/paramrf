import jax.numpy as jnp

from pmrf.sampling.oneshot import OneshotSampler
from pmrf.util import lhs_sample

class LatinHypercubeSampler(OneshotSampler):
    """
    Sampler using Latin Hypercube Sampling (LHS).

    LHS is a stratified sampling method that generates sample points that are
    more evenly distributed across the hypercube than standard random sampling.
    This implementation uses `scipy.stats.qmc.LatinHypercube`.
    """
    def _generate(self, N: int, d: int, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate samples using Latin Hypercube Sampling.

        Parameters
        ----------
        N : int
            Number of samples.
        D : int
            Dimensionality (number of parameters).
        key: jnp.ndarray
            The JAX key,

        Returns
        -------
        jnp.ndarray
            Samples in the unit hypercube `[0, 1)^D`.
        """
        return lhs_sample(N, d, key)