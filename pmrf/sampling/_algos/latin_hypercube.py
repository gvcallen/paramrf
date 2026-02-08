import jax
import jax.numpy as jnp

from pmrf.sampling.oneshot import OneshotSampler

class LatinHypercubeSampler(OneshotSampler):
    """
    Sampler using Latin Hypercube Sampling (LHS).

    LHS is a stratified sampling method that generates sample points that are
    more evenly distributed across the hypercube than standard random sampling.
    This implementation uses `scipy.stats.qmc.LatinHypercube`.
    """
    def _generate(self, N: int, d: int, key=None) -> jnp.ndarray:
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
        key_perm, key_noise = jax.random.split(key)
        keys_perm = jax.random.split(key_perm, d)
        perms = jax.vmap(lambda k: jax.random.permutation(k, N))(keys_perm)
        noise = jax.random.uniform(key_noise, shape=(d, N))
        lhs_unit = (perms + noise) / N
        lhs_unit = lhs_unit.T
        
        return lhs_unit