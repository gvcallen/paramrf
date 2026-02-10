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
    # The coefficients parameters (coefficients)
    coefficients_real: Parameter
    
    # The basis functions themselves and an optional offset
    basis: jnp.ndarray
    
    coefficients_imag: Parameter | None = None
    offset: jnp.ndarray | None = None
    
    @property
    def num_basis(self) -> int:
        return len(self.basis)
    
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
                basisi = jnp.real(basisi)
            elif component == 'im':
                basisi = jnp.imag(basisi)
            elif component == 'db':
                basisi = 20*jnp.log10(jnp.abs(basisi))
            
            axes[i].plot(self.frequency.f_scaled, basisi)
            axes[i].set_title(f'Basis {i}')

        fig.set_size_inches(10, 6)
        fig.tight_layout()

        return fig
    
    def plot_basis_db(self):
        self.plot_basis('db')
        
    def plot_basis_mag(self):
        self.plot_basis('mag')

    def plot_basis_re(self):
        self.plot_basis('re')
        
    def plot_basis_im(self):
        self.plot_basis('im')
        
    def plot_error(self, samples: jnp.ndarray, component):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        for i in range(len(samples)):
            measured = samples[i]
            coeff = self.inverse(measured)
            model = self.with_params(coeff).forward()
            
            error = model - measured
            error = error[:,0,0]
            if component == 'mag':
                error = jnp.abs(error)
            elif component == 're':
                error = jnp.real(error)
            elif component == 'im':
                error = jnp.imag(error)
            elif component == 'db':
                error = 20*jnp.log10(jnp.abs(error))
            
            ax.plot(self.frequency.f_scaled, error)
        
        ax.set_title(f'Sample Reprojection Error (num_basis = {self.num_basis})')
        ax.set_xlabel(f'Frequency [{self.frequency.unit}]')
        ax.set_ylabel(f'Error [{component}]')
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        return fig
    
    def plot_error_db(self, samples):
        self.plot_error(samples, 'db')
        
    def plot_error_mag(self, samples):
        self.plot_error(samples, 'mag')
    
    def plot_error_re(self, samples):
        self.plot_error(samples, 're')
    
    def plot_error_im(self, samples):
        self.plot_error(samples, 'im')
        
        
class SVDExpansion(BasisExpansion):
    @classmethod
    def from_samples(cls, features: jnp.ndarray, frequency: Frequency, property='s', min_components=1, max_components=1000, error_threshold=None) -> Self:
        """
        Creates an SVD expansion basis from samples with arbitrary dimensions.
        
        Args:
            features: Shape (nsamples, nfreq, m, n)
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

        if error_threshold is not None:
            for n_components in range(min_components, max_components + 1):
                current_comps = Vh[..., :n_components, :]
                current_coeff = jnp.einsum('mnbf,mnkf->mnbk', Xc, jnp.conj(current_comps))
                Xc_reproj = jnp.einsum('mnsc,mncf->mnsf', current_coeff, current_comps)
                X_reproj = Xc_reproj + X_mean
                X_error = jnp.abs(X - X_reproj)
                if jnp.max(X_error) < error_threshold:
                    break
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