from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler

class OneshotSampler(BaseSampler, ABC):
    """
    Base class for samplers that propose all their points at once 
    (e.g., Uniform, LHS, Sobol).
    """
    def sample(
        self,
        *,
        N: int = 100,
        **kwargs
    ) -> tuple[jnp.ndarray, Any]:
        d = self.model.num_flat_params

        U = self.generate(N, d, **kwargs)
        thetas = jnp.array([self.icdf(u) for u in U])
        self.update(thetas)
        return self.sampled_params, None

    @abstractmethod
    def generate(self, N: int, d: int, **kwargs) -> jnp.ndarray:
        """Return an (N, d) array of points in the unit hypercube [0, 1]."""
        raise NotImplementedError