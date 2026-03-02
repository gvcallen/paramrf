from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler
from pmrf.models.model import Model
from pmrf.util import lhs_sample

class AcquisitionSampler(BaseSampler, ABC):
    r"""
    Base class for adaptive, acquisition-based sampling (Active Learning).
    
    This sampler implements a feedback loop that:
    
    * Starts with an initial random experimental design (LHS).
    * Repeatedly queries an acquisition function to find the most "informative" 
      next points to simulate.
    * Updates the internal state until a budget ($N$) or convergence is reached.

    This class is marked as ``expensive=True``, meaning it will automatically 
    save intermediate NumPy checkpoints for crash recovery during the loop.

    This class is marked as ``expensive=True``, meaning it will automatically 
    save intermediate NumPy checkpoints for crash recovery during the loop.

    .. rubric:: Methods

    .. autosummary::
       :nosignatures:

       run
       sample
       update
       acquire
    """
    expensive: bool = True
    
    def execute(
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
        r"""
        Execute the adaptive active learning loop.
        
        **NB:** This method should not be called directly. Call :meth:`run` instead.

        Parameters
        ----------
        N : int, optional
            The total budget of samples. The loop stops once ``len(sampled_params) >= N``.
        initial_models : list of Model or int, optional
            The starting state. If an integer, that many points are sampled via 
            Latin Hypercube Sampling (LHS) to start the loop.
        initial_factor : int, optional
            A multiplier for the parameter count to determine the number of 
            initial points (e.g., ``initial_factor=5`` for a 2-param model 
            starts with 10 points).
        batch_size : int, default=1
            The number of points to propose and simulate in each iteration.
        max_iterations : int, optional
            The maximum number of adaptive loops to run, regardless of the 
            total sample count.
        key : jax.Array, optional
            A JAX random PRNG key for stochastic acquisition functions.
        **kwargs
            Additional arguments passed to the :meth:`acquire` method.

        Returns
        -------
        tuple[jax.numpy.ndarray, None]
            The final array of sampled physical parameters.

        Raises
        ------
        ValueError
            If initialization arguments are contradictory or insufficient.
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
            self.update(batch_theta)
        
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
                self.update(batch_theta)
            
            # 5. Check Stopping Criteria
            if N is not None and len(self.sampled_params) >= N:
                break
            
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                self.logger.warning("Maximum iterations were reached during adaptive sampling.")
                break            
            
        return self.sampled_params, None
    
    @abstractmethod
    def acquire(self, N: int, d: int, *, key: jax.Array | None = None, **kwargs) -> jnp.ndarray | None:
        """
        Implemented by active learning backends (e.g., EqxLearnUncertaintySampler).
        
        Should inspect `self.sampled_params` and `self.sampled_features`, train 
        a surrogate/acquisition function, and return `N` proposed points in the 
        unit hypercube (shape `(N, d)`).
        
        Return `None` to signal algorithmic convergence to the master loop.
        """
        raise NotImplementedError