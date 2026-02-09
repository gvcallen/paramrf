import jax.numpy as jnp
import equinox as eqx

from pmrf.models.blackbox.blackbox import BlackBox, SupervisedBlackBox
from pmrf.parameters import Parameter
from pmrf.models import Model

class LatentProjection(SupervisedBlackBox):
    """
    A model that computes its output via projection into a latent space.
    
    The output is calculated using a encoder-decoder architecture:
    - The model is a function of D parameters.
    - The encoder is an arbitrary equinox Module with D inputs and K "latent" outputs.
    - The decoder is a ParamRF model with K parameters.
    """
    # The latent encoder. Must be callable with D inputs and must have K outputs
    encoder: eqx.Module = eqx.field(static=True)
    
    # The latent decoder. This must have D flat parameters
    decoder: BlackBox = eqx.field(static=True)
    
    # The current parameters of length P
    params: Parameter
    
    def forward(self) -> jnp.ndarray:
        # The forward model, which produces a sample for the current parameters
        return self.decoder.with_params(self.encoder(self.flat_param_values())).forward()