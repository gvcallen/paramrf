import jax.numpy as jnp
import equinox as eqx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency
from pmrf.core.evaluator import Evaluator
from pmrf.distributions import JointDistribution

class Problem(eqx.Module):
    model: Model
    frequency: Frequency
    evaluator: Evaluator
    
    def __call__(self) -> jnp.ndarray:
        return self.evaluator(self.model, self.frequency)
    
    def distribution(self, param_groups=True) -> JointDistribution:
        model_dist = self.model.distribution(param_groups=param_groups)
        evaluator_dist = 
d