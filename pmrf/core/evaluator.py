import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency

class Evaluator(prx.Module):
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError