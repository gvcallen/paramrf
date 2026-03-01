from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler
from pmrf.util import lhs_sample

class OneshotSampler(BaseSampler, ABC):
    """
    Base class for samplers that propose all their points at once 
    (e.g., Uniform, LHS, Sobol).
    """
    def sample(self, N: int, *, plot=None, **kwargs) -> tuple[jnp.ndarray, Any]:
        d = self.model.num_flat_params

        U = self.generate(N, d, **kwargs)
        thetas = jnp.array([self.icdf(u) for u in U])
        self.add_samples(thetas, plot=plot)
        return self.sampled_params, None

    @abstractmethod
    def generate(self, N: int, d: int, **kwargs) -> jnp.ndarray:
        """Return an (N, d) array of points in the unit hypercube [0, 1]."""
        raise NotImplementedError