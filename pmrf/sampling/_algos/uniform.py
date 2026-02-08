import jax.numpy as jnp
import jax

from pmrf.sampling.oneshot import OneshotSampler

class UniformSampler(OneshotSampler):
    def _generate(self, N, D, key=None) -> jnp.ndarray:
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
        return jax.random.uniform(key, shape=(N, D))