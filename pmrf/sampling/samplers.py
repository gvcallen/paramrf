import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from pmrf.sampling._base import BaseSampler

class LatinHypercubeSampler(BaseSampler):
    def _generate_hypercube_samples(self, N, D) -> jnp.ndarray:
        return jnp.array(qmc.LatinHypercube(D).random(N))

class UniformSampler(BaseSampler):
    def _generate_hypercube_samples(self, N, D) -> jnp.ndarray:
        return jnp.array(np.random.uniform(0.0, 1.0, size=(N, D)))