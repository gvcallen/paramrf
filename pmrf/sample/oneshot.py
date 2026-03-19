from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp

from pmrf.sample.base import BaseSampler
from pmrf.utils import generate_key

class OneshotSampler(BaseSampler, ABC):
    r"""
    Base class for static, non-adaptive sampling strategies.
    
    This class is used for strategies that propose all sample points at once, 
    independent of any model feedback (e.g., Uniform, Latin Hypercube, or Sobol 
    sequences). It handles the transformation from the unit hypercube to the 
    physical parameter space using the model's prior distributions.

    .. rubric:: Methods

    .. autosummary::
       :nosignatures:

       run
       execute
       generate
    """
    def execute(
        self,
        *,
        N: int = 100,
        batch_size: int = 1,
        key=None,
        **kwargs
    ) -> tuple[jnp.ndarray, Any]:
        r"""
        Generate and evaluate a batch of points in one execution.

        Parameters
        ----------
        N : int, default=100
            The total number of samples to generate and simulate.
        batch_size : int, default=1
            The number of points to simulate in each iteration.            
        **kwargs
            Additional arguments passed to the :meth:`generate` method.

        Returns
        -------
        tuple[jax.numpy.ndarray, None]
            A tuple containing the evaluated physical parameters and a 
            placeholder for backend results.
        """
        if key is None:
            key = generate_key()

        d = self.model.num_flat_params

        U = self.generate(N, d, key=key, **kwargs)
        thetas = jnp.array([self.icdf(u) for u in U])
        
        for i in range(0, thetas.shape[0], batch_size):
            batch_theta = thetas[i : i + batch_size]
            self.update(batch_theta)
        
        return self.sampled_params, None

    @abstractmethod
    def generate(self, N: int, d: int, **kwargs) -> jnp.ndarray:
        r"""
        Propose points within the unit hypercube $[0, 1]^d$.

        Parameters
        ----------
        N : int
            Number of points to propose.
        d : int
            Dimensionality of the parameter space.

        Returns
        -------
        jax.numpy.ndarray
            An array of shape ``(N, d)`` containing the proposed coordinates.
        """
        raise NotImplementedError