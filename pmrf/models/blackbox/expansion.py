from typing import Self

import jax.numpy as jnp
from jax.scipy.linalg import svd

from pmrf.frequency import Frequency
from pmrf._util import field
from pmrf.models.blackbox.blackbox import UnsupervisedBlackBox
from pmrf.parameters import Parameter, Uniform

class BasisExpansion(UnsupervisedBlackBox):
    """An RF model where the S-parameters are modeled as a linear expansion of basis functions.
    The S-parameters are returned as offset + coefficients @ basis, where the coefficients are the parameters.
    """
    # The parameters (coefficients)
    coefficients_real: Parameter
    coefficients_imag: Parameter | None = None
    
    # The basis functions themselves and an optional offset
    basis: jnp.ndarray = field(default_factory=lambda: 0.0, static=True)
    offset: jnp.ndarray | None = field(default_factory=lambda: 0.0, static=True)
    
    @property
    def coefficients_complex(self) -> jnp.ndarray:
        coefficients = self.coefficients_real
        if self.coefficients_imag is not None:
            coefficients += 1j * self.coefficients_imag
        return coefficients    
    
    def forward(self) -> jnp.ndarray:
        # The forward model, which multiplies the current coefficients onto the basis vectors
        coeff = self.coefficients_complex
        X = jnp.einsum('imn,ikmn->kmn', coeff, self.basis)
        
        if self.offset is not None:
            offset = self.offset.reshape(X.shape)
            X += offset

        return X
    
    def inverse(self, sample: jnp.ndarray) -> jnp.ndarray:
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
    
    def plot_basis(self, component, m=0, n=0):
        import matplotlib.pyplot as plt

        basis = self.basis
        nbasis = len(basis)
        rows = int(jnp.sqrt(nbasis))
        cols = int(nbasis / rows) + 1
        fig, axes = plt.subplots(rows, cols)
        axes = axes.flatten()
        for i in range(nbasis):
            basisi = basis[i,:,m,n]
            
            if component == 'mag':
                basisi = jnp.abs(basisi)
            elif component == 're':
                basisi = jnp.imag(basisi)
            elif component == 'im':
                basisi = jnp.real(basisi)
            
            axes[i].plot(self.frequency.f_scaled, basisi)
            axes[i].set_title(f'Basis {i}')

        fig.set_size_inches(10, 6)
        fig.tight_layout()

        return fig
    
    def plot_basis_mag(self):
        self.plot_basis('mag')

    def plot_basis_re(self):
        self.plot_basis('re')
        
    def plot_basis_im(self):
        self.plot_basis('im')
        
class SVDExpansion(BasisExpansion):
    @classmethod
    def from_samples(cls, features: jnp.ndarray, frequency: Frequency, property='s', min_components=1, max_components=100, var_threshold=None) -> Self:
        """
        Creates an SVD expansion basis from samples with arbitrary dimensions.
        
        Args:
            samples: Shape (nsamples, nfreq, m, n)
        Returns:
            SVDExpansion with basis shape (ncomponents, nfreq, m, n)
        """
        nsamples, nfreq, m, n = features.shape
        X = jnp.transpose(features, (2, 3, 0, 1))
        X_mean = jnp.mean(X, axis=2, keepdims=True)
        Xc = X - X_mean
        full_max_components = min(nsamples, nfreq)
        _Uh, _s, Vh = svd(Xc, full_matrices=False)

        max_components = min(max_components, full_max_components)
        min_components = min(min_components, full_max_components)
        n_components = min_components

        if var_threshold is not None:
            total_variance = jnp.var(Xc, axis=2).sum(axis=-1)
            for k in range(min_components, max_components + 1):
                current_comps = Vh[..., :k, :]
                Z = jnp.einsum('mnbf,mnkf->mnbk', Xc, jnp.conj(current_comps))

                # Explained variance is the variance of the projections
                explained_variance = jnp.var(Z, axis=2).sum(axis=-1)
                ratios = explained_variance / total_variance

                if jnp.min(ratios) >= var_threshold:
                    n_components = k
                    break
                
                n_components = k
        else:
            n_components = min_components

        # Select final components: (m, n, n_components, nfreq)
        components = Vh[..., :n_components, :]

        # Calculate coefficients for the trained model
        # Shape: (m, n, nsamples, n_components)
        coeffs = jnp.einsum('mnbf,mnkf->mnbk', Xc, jnp.conj(components))

        # Calculate Ranges (Real/Imag)
        coeffs_real = jnp.real(coeffs)
        coeffs_imag = jnp.imag(coeffs)
        
        c_real_min = jnp.min(coeffs_real, axis=2)
        c_real_max = jnp.max(coeffs_real, axis=2)
        c_imag_min = jnp.min(coeffs_imag, axis=2)
        c_imag_max = jnp.max(coeffs_imag, axis=2)

        basis_out = jnp.transpose(components, (2, 3, 0, 1))
        offset_out = jnp.transpose(X_mean, (2, 3, 0, 1))
        c_real_min = jnp.transpose(c_real_min, (2, 0, 1))
        c_real_max = jnp.transpose(c_real_max, (2, 0, 1))
        c_imag_min = jnp.transpose(c_imag_min, (2, 0, 1))
        c_imag_max = jnp.transpose(c_imag_max, (2, 0, 1))

        return SVDExpansion(
            coefficients_real=Uniform(c_real_min, c_real_max),
            coefficients_imag=Uniform(c_imag_min, c_imag_max),
            basis=basis_out,
            offset=offset_out,
            frequency=frequency,
            property=property,
        )