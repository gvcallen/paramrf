import jax.numpy as jnp
import equinox as eqx

from pmrf.model import Model
from pmrf.frequency import Frequency

class Evaluator(eqx.Module):
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError