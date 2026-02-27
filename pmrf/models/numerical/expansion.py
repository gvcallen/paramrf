import jax.numpy as jnp

from pmrf.models.numerical.base import Numerical
from pmrf.models.adapter import SingleDiscreteProperty
from pmrf.parameters import Parameter

class VectorExpansion(Numerical, SingleDiscreteProperty):
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