from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.extractor import Extractor
from pmrf.metrics import root_mean_squared_error

class Objective(eqx.Module):
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError

class Residual(Objective):
    extractor: Extractor
    condition: str = eqx.field(default='==', static=True) # '>', '==', or '<'
    target: float | jnp.ndarray
    threshold_fn: Callable | tuple[Callable, Callable] = eqx.field(default=(jnp.minimum, jnp.maximum), static=True)
    metric_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = eqx.field(default=root_mean_squared_error, static=True)
    
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        prediction = self.extractor(model, freq)
        
        if self.condition == '==':
            pass
        elif self.condition == '<':
            prediction = self.threshold_fn[1](prediction, self.target)
        elif self.condition == '>':
            prediction = self.threshold_fn[0](prediction, self.target)
        else:
            raise ValueError(f"Unknown condition: {self.condition}")        
        
        return self.metric_fn(self.target, prediction)
    
class Flatness(Objective):
    extractor: Extractor
    metric_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = eqx.field(default=root_mean_squared_error, static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        prediction = self.extractor(model, freq)
        variation = jnp.gradient(prediction, freq.f_scaled) 
        target = jnp.zeros_like(variation)
        return self.metric_fn(target, variation)
    
    
class Regularization(Objective):
    param_filter: str  # e.g., 'ind.*'
    penalty_fn: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        param_values = model.param_values(self.param_filter)
        return jnp.concatenate([self.penalty_fn(v) for v in param_values])