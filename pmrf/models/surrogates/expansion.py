"""
An expansion of a set of basis functions.
"""

import jax.numpy as jnp

from pmrf.models.adapters.base import AbstractSingleDiscreteDomain
from pmrf.types import Param
from pmrf.parameters import param

class VectorExpansion(AbstractSingleDiscreteDomain):
    """
    (experimental) A model where the output is a linear expansion of vector/matrix basis functions with an optional offset.
    
    The S-parameters are returned as offset + coefficients @ basis, where the coefficients are the model parameters.
    
    Parameters
    ----------
    coefficients_real : Param
        The real coefficients
    coefficients_imag : Param
        The imaginary coefficients
    basis : Param
        The basis functions
    offset : Param
        An optional offset
    """
    #: The real coefficients parameters
    coefficients_real: Param = param()
    
    #: The imaginary coefficients parameters
    coefficients_imag: Param = param()
    
    #: The basis functions
    basis: Param = param()
    
    #: An optional fixed offset
    offset: Param = param()
    
    def discrete_matrix(self) -> jnp.ndarray:
        coeff = self.coefficients_real
        if self.coefficients_imag is not None:
            coeff += 1j * self.coefficients_imag
        
        # The model output which multiplies the current coefficients onto the basis vectors
        X = jnp.einsum('imn,ikmn->kmn', coeff, self.basis)
        
        if self.offset is not None:
            offset = self.offset.reshape(X.shape)
            X += offset

        return X
    
    def inverse(self, sample: jnp.ndarray) -> jnp.ndarray:
        # TODO deprecate this method and introduce
        # the inverse solution in the "solver module

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