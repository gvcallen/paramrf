import equinox as eqx
import jax.numpy as jnp

from pmrf.models.model import Model
from pmrf.parameters import Parameter

class EquinoxBlackBox(Model):
    """
    A BlackBox model represented by an Equinox module that takes in the flat parameters as an input.
    The Equinox module must accept these parameters in __call__ and must output an array corresponding to ``self.frequency``.
    """
    params: Parameter
    model: eqx.Module
    
    def forward(self) -> jnp.ndarray:
        if not hasattr(self.model, '__call__'):
            raise Exception("The Equinox model in EquinoxBlackBox must have a __call__ method for the forward pass")
        
        return self.model(self.flat_param_values())