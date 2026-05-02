"""
A callable to be "solved" (i.e. minimized or sampled).
"""

import jax.numpy as jnp
import equinox as eqx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency
from pmrf.core.evaluator import Evaluator

class Problem(eqx.Module):
    """
    A callable to be "solved" (i.e. minimized or sampled).
    
    This class encapsulates a model, its frequency domain, and the 
    evaluator (such as a loss or likelihood) into a single callable unit.
    """
    
    #: The RF model to be evaluated.
    model: Model
    
    #: The frequency range or points over which the model is evaluated.
    frequency: Frequency
    
    #: The operator (e.g., a Likelihood or Loss) that maps the model 
    #: and frequency to a scalar or array result.
    evaluator: Evaluator
    
    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        """
        Call the evaluator on the model and frequency.

        Returns
        -------
        jnp.ndarray
            The result of evaluating the model with the stored evaluator and frequency.
        """        
        return self.evaluator(self.model, self.frequency, *args, **kwargs)