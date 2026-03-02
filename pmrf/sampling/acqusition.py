from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler
from pmrf.models.model import Model
from pmrf.util import lhs_sample

class AcquisitionSampler(BaseSampler, ABC):
    """
    Base class for acquisition-based adaptive sampling strategies.
    
    This class implements the `sample()` algorithm by repeatedly querying 
    a concrete subclass's `acquire()` method until convergence or limits are reached.
    """
    def sample(
        self, 
        N: int | None = None, 
        *, 
        initial_models: list[Model] | int | None = None,
        initial_factor: int | None = None,
        batch_size: int = 1, 
        max_iterations: int | None = None, 
        key: jax.Array | None = None, 
        **kwargs
    ) -> tuple[jnp.ndarray, Any]:
        """
        Executes the active learning loop.
        """
        # 1. Parse Initial Setup Args
        if initial_factor is not None and initial_models is not None:
            raise ValueError("Cannot pass both initial_factor and initial_models.")
        if initial_factor is not None:
            initial_models = initial_factor * self.model.num_flat_params
            
        if isinstance(initial_models, int) and initial_models < 2:
            raise ValueError("Number of initial models must be at least 2.")
            
        initial_models_list = list(initial_models) if not isinstance(initial_models, int) else initial_models        

        if key is None:
            key = acquire_key()
            
        d = self.model.num_flat_params

        # 2. Generate Initial Random Sampling Phase
        if isinstance(initial_models_list, int):
            key, initial_key = jr.split(key)
            initial_Us = lhs_sample(initial_models_list, d, key=initial_key)
            # Use lazy base runner accessor
            initial_thetas = jax.vmap(self.icdf)(initial_Us)
            initial_models_list = [self.model.with_params(theta) for theta in initial_thetas]
        
        # We explicitly extract params to a numpy array for standard processing
        initial_thetas = jnp.array([m.flat_param_values() for m in initial_models_list])
        num_initial_samples = len(initial_thetas)
        
        # 3. Add initial samples in batches (updates internal state)
        for i in range(0, num_initial_samples, batch_size):
            batch_theta = initial_thetas[i : i + batch_size]
            self.add_samples(batch_theta)
        
        # 4. The Active Learning Loop
        iteration = 0
        while True:
            key, acquire_key = jr.split(key)
            
            # The backend strategy determines the next points to evaluate
            U_next = self.acquire(batch_size, d, key=acquire_key, **kwargs)
            
            if U_next is None:
                self.logger.info("Sampling converged.")
                break
                
            # Transform hypercube proposed points back to physical parameters
            thetas = jnp.array([self.icdf(u) for u in U_next])
            num_samples = len(thetas)
            
            # Add to state and evaluate features
            for i in range(0, num_samples, batch_size):
                batch_theta = thetas[i : i + batch_size]
                self.add_samples(batch_theta)
            
            # 5. Check Stopping Criteria
            if N is not None and len(self.sampled_params) >= N:
                break
            
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                self.logger.warning("Maximum iterations were reached during adaptive sampling.")
                break            
            
        # Returning (samples, backend_state). Adaptive Samplers usually don't 
        # have specific backend objects to save, so we return None for the state.
        return self.sampled_params, None
    
    @abstractmethod
    def acquire(self, N: int, d: int, *, key: jax.Array | None = None, **kwargs) -> jnp.ndarray | None:
        """
        Implemented by active learning backends (e.g., EqxLearnSurrogateSampler).
        
        Should inspect `self.sampled_params` and `self.sampled_features`, train 
        a surrogate/acquisition function, and return `N` proposed points in the 
        unit hypercube (shape `(N, d)`).
        
        Return `None` to signal algorithmic convergence to the master loop.
        """
        raise NotImplementedError