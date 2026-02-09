import jax.numpy as jnp
import equinox as eqx

from pmrf.models.blackbox.blackbox import BlackBox, SupervisedBlackBox
from pmrf.parameters import Parameter

class LatentProjection(SupervisedBlackBox):
    """
    A model that computes its output via projection into a latent space.
    
    The output is calculated using a encoder-decoder architecture:
    - The model is a function of D parameters.
    - The encoder is an arbitrary equinox Module with D inputs and K "latent" outputs.
    - The decoder is a ParamRF model with K parameters.
    """
    # The current parameters of length P
    params: Parameter
    
    # The latent encoder. Must be callable with D inputs and must have K outputs
    encoder: eqx.Module
    
    # The latent decoder. This must have D flat parameters
    decoder: BlackBox
    
    def forward(self) -> jnp.ndarray:
        # The forward model, which produces a sample for the current parameters
        return self.decoder.with_params(self.encoder(self.flat_param_values())).forward()