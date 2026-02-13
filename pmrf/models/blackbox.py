import jax.numpy as jnp
import equinox as eqx

from pmrf.models.abstract import Model, InterpolatedSingleProperty
from pmrf.parameters import Parameter

class Latent(Model):
    """
    A model that computes its output via a latent space.
    
    The output is calculated using a encoder-decoder architecture:
    - The model is a function of D parameters.
    - The encoder is an arbitrary equinox Module with D inputs and K "latent" outputs.
    - The decoder is a ParamRF model with K parameters.
    """
    # The input parameters of length P
    params: Parameter = None
    
    # The latent encoder. Must be callable with D inputs and must have K outputs
    encoder: eqx.Module = None
    
    # The latent decoder. This must have K flat parameters
    decoder: Model = None
    
    def __post_init__(self):
        self.decoder = self.decoder.with_all_params_fixed()
    
    def __call__(self) -> Model:
        # The forward model, which produces a sample for the current parameters
        return self.decoder.with_all_params_free().with_params(self.encoder(self.flat_param_values()))

class VectorExpansion(InterpolatedSingleProperty):
    """
    A model where the output is a linear expansion of vector basis functions.
    
    The S-parameters are returned as offset + coefficients @ basis, where the coefficients are the model parameters.
    """
    # The coefficients parameters (coefficients)
    coefficients_real: Parameter = None
    coefficients_imag: Parameter = None
    
    # The basis functions themselves and an optional offset
    basis: jnp.ndarray = None
    offset: jnp.ndarray = None
    
    def output_discrete(self) -> jnp.ndarray:
        # The model output which multiplies the current coefficients onto the basis vectors
        coeff = self.coefficients_complex
        X = jnp.einsum('imn,ikmn->kmn', coeff, self.basis)
        
        if self.offset is not None:
            offset = self.offset.reshape(X.shape)
            X += offset

        return X
    
    def inverse(self, sample: jnp.ndarray) -> jnp.ndarray:
        if len(sample.shape) == 1:
            sample = sample[..., None, None]
        
        # The inverse model, which projects a sample onto the coefficients
        if self.offset is not None:
            sample = sample - self.offset
        basis_Tconj = self.basis.transpose(1, 0, 2, 3).conj()
        
        # This projects the sample onto the basis vector for each port (m, n)
        coefficients = jnp.einsum('ifmn,fbmn->ibmn', sample, basis_Tconj).reshape(basis_Tconj.shape[1:])

        if self.coefficients_imag is not None:
            coefficients = jnp.concat([coefficients.real, coefficients.imag])
        else:
            coefficients = coefficients.real
        return coefficients    
    
    @property
    def num_basis(self) -> int:
        return len(self.basis)
    
    @property
    def basis_separate(self) -> jnp.ndarray:
        return jnp.concatenate([self.basis.real, self.basis.imag], axis=0)
    
    @property
    def coefficients_complex(self) -> jnp.ndarray:
        coefficients = self.coefficients_real
        if self.coefficients_imag is not None:
            coefficients += 1j * self.coefficients_imag
        return coefficients    