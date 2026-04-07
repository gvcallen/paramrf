import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency
from pmrf.core.evaluator import Evaluator

class Problem(prx.Module):
    """
    A callable to be "solved" (i.e. minimized or sampled).
    
    This class encapsulates a model, its frequency domain, and the 
    evaluator (such as a loss or likelihood) into a single callable unit.

    Attributes
    ----------
    model : Model
        The RF model to be evaluated.
    frequency : Frequency
        The frequency range or points over which the model is evaluated.
    evaluator : Evaluator
        The operator (e.g., a Likelihood or Loss) that maps the model 
        and frequency to a scalar or array result.
    """
    
    model: Model
    frequency: Frequency
    evaluator: Evaluator
    
    def __call__(self) -> jnp.ndarray:
        """
        Execute the problem evaluation.

        Returns
        -------
        jnp.ndarray
            The result of evaluating the model with the stored evaluator and frequency.
        """        
        return self.evaluator(self.model, self.frequency)