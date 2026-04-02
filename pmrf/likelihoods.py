from typing import Callable

import jax
import jax.numpy as jnp
import distreqx.distributions as dist
import distreqx.bijectors as bij
import parax as prx

from pmrf.bijectors import RealToComplex, Rotate, LogPolarToComplex, MoveAxis
from pmrf.distributions import MultivariateNormalFullCovariance
from pmrf.core import Likelihood


def fix_sigma_shape(sigma, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Broadcasts a sigma parameter based on the shape of y_pred.
    
    If y_pred is of shape (nfreq, n, n) or (nfreq, n, n, ...), it attempts to 
    broadcast scalar or 2-element sigmas into an (n, n) matrix. It automatically
    pads trailing ones (n, n, 1, ...) to ensure safe tensor broadcasting.
    
    Otherwise, it returns sigma directly, assuming the caller provided a properly 
    broadcastable array.
    """
    sigma = jnp.asarray(sigma)
    
    # Check for port matrix structure: at least 3 dims, and axis 1 == axis 2
    if y_pred.ndim >= 3 and y_pred.shape[1] == y_pred.shape[2]:
        n = y_pred.shape[1]
        sigma_matrix = None
        
        if sigma.size == 1:
            sigma_matrix = jnp.full((n, n), sigma.squeeze())
        elif sigma.size == 2:
            sigma_sq = sigma.squeeze()
            sigma_matrix = jnp.full((n, n), sigma_sq[1])
            diag_indices = jnp.diag_indices(n)
            sigma_matrix = sigma_matrix.at[diag_indices].set(sigma_sq[0])
        elif sigma.shape == (n, n) or sigma.size == n ** 2:
            sigma_matrix = sigma.reshape((n, n))
            
        if sigma_matrix is not None:
            # Pad with trailing ones for safe right-to-left broadcasting against y_pred
            # e.g., if y_pred is (nfreq, n, n, k, j), pad_shape becomes (n, n, 1, 1)
            num_trailing_dims = y_pred.ndim - 3
            if num_trailing_dims > 0:
                pad_shape = (n, n) + (1,) * num_trailing_dims
                sigma_matrix = sigma_matrix.reshape(pad_shape)
                
            return sigma_matrix
            
    # Fallback: assume the caller provided a correctly shaped/broadcastable array
    return sigma


class GaussianLikelihood(Likelihood):
    """Standard Gaussian for purely real-valued network features."""
    sigma: prx.Parameter | jnp.ndarray | float

    def __call__(self, y_pred: jnp.ndarray, **kwargs):
        sigma_matrix = fix_sigma_shape(self.sigma, y_pred)
        return dist.Normal(loc=y_pred, scale=sigma_matrix)


class ComplexGaussianLikelihood(Likelihood):
    """Complex Gaussian where variance is split evenly between Real and Imaginary."""
    sigma: prx.Parameter | jnp.ndarray | float

    def __call__(self, y_pred: jnp.ndarray, **kwargs):
        sigma_matrix = fix_sigma_shape(self.sigma, y_pred)
        
        # Split variance evenly
        sigma_parts = sigma_matrix / jnp.sqrt(2.0)
        scale_diag = jnp.stack([sigma_parts, sigma_parts], axis=-1)
        
        # R^2 Prediction Location
        loc = jnp.stack([jnp.real(y_pred), jnp.imag(y_pred)], axis=-1)
        
        base_dist = dist.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        return dist.Transformed(base_dist, RealToComplex())


class MagnitudePhaseGaussianLikelihood(Likelihood):
    """Models relative magnitude and absolute phase noise independently."""
    sigma_mag: prx.Parameter | jnp.ndarray | float
    sigma_phase: prx.Parameter | jnp.ndarray | float

    def __call__(self, y_pred: jnp.ndarray, **kwargs):
        sig_m = fix_sigma_shape(self.sigma_mag, y_pred)
        sig_p = fix_sigma_shape(self.sigma_phase, y_pred)
        
        scale_diag = jnp.stack([sig_m, sig_p], axis=-1)
        
        # R^2 Location in Log-Polar space
        log_mag_pred = jnp.log(jnp.abs(y_pred) + 1e-12)
        phase_pred = jnp.angle(y_pred)
        loc = jnp.stack([log_mag_pred, phase_pred], axis=-1)
        
        base_dist = dist.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        
        # The LogPolar bijector naturally wraps the phase when transitioning to complex
        return dist.Transformed(base_dist, LogPolarToComplex())


class RadialTangentialGaussianLikelihood(Likelihood):
    """Geometrically aligns noise to the prediction vector (Banana shape)."""
    sigma_complex: prx.Parameter | jnp.ndarray | float
    sigma_mag: prx.Parameter | jnp.ndarray | float
    sigma_phase: prx.Parameter | jnp.ndarray | float

    def __call__(self, y_pred: jnp.ndarray, **kwargs):
        sig_c = fix_sigma_shape(self.sigma_complex, y_pred)
        sig_m = fix_sigma_shape(self.sigma_mag, y_pred)
        sig_p = fix_sigma_shape(self.sigma_phase, y_pred)
        
        # 1. Heteroscedastic Scales
        mag_pred = jnp.abs(y_pred)
        var_base = (sig_c ** 2) / 2.0
        var_rad = var_base + (mag_pred * sig_m) ** 2
        var_tan = var_base + (mag_pred * sig_p) ** 2
        
        scale_diag = jnp.sqrt(jnp.stack([var_rad, var_tan], axis=-1))

        # 2. Base Distribution at the Origin (Unrotated)
        base_dist = dist.MultivariateNormalDiag(
            loc=jnp.zeros_like(scale_diag), 
            scale_diag=scale_diag
        )

        # 3. Geometric Transforms
        # A. Rotate the (rad, tan) error to align with the prediction's phase
        phase_pred = jnp.angle(y_pred)
        rotate_bij = Rotate(angle=phase_pred)
        
        # B. Shift the rotated error onto the prediction
        y_pred_reim = jnp.stack([jnp.real(y_pred), jnp.imag(y_pred)], axis=-1)
        shift_bij = bij.Shift(shift=y_pred_reim)
        
        # 4. Compose Chain: R^2(Origin) -> R^2(Rotated) -> R^2(Shifted) -> Complex
        bijector_chain = bij.Chain([RealToComplex(), shift_bij, rotate_bij])
        
        return dist.Transformed(base_dist, bijector_chain)
    
    
class GaussianProcessLikelihood(Likelihood):
    """
    Wraps a base likelihood and injects a Gaussian Process covariance 
    structure into its base coordinate space across the frequency axis.
    """
    base_likelihood: Likelihood
    
    # Expects **kwargs (contains 'frequency') and returns covariance matrix of shape (nfreq, nfreq)
    kernel: Callable[..., jnp.ndarray] 

    def __call__(self, y_pred: jnp.ndarray, **kwargs) -> dist.AbstractDistribution:
        # 1. Base independent distribution
        base_dist = self.base_likelihood(y_pred, **kwargs)

        # 2. Extract Bijectors & Foundation
        bijectors = []
        current_dist = base_dist
        
        while isinstance(current_dist, dist.Transformed):
            bijectors.append(current_dist.bijector)
            current_dist = current_dist.distribution
            
        if not isinstance(current_dist, (dist.Normal, dist.MultivariateNormalDiag)):
            raise TypeError("GaussianProcessLikelihood requires a Gaussian base likelihood.")

        # 3. Extract parameters
        # Shape: (nfreq, d1, d2, ..., 2)
        variances = current_dist.scale ** 2 if isinstance(current_dist, dist.Normal) else current_dist.scale_diag ** 2
        loc = current_dist.loc
        nfreq = loc.shape[0]

        # 4. Evaluate GP Kernel (Shape: nfreq, nfreq)
        gp_covariances = self.kernel(**kwargs)

        # 5. Move Frequency to the Event Dimension
        # Shape: (d1, d2, ..., 2, nfreq)
        loc_aligned = jnp.moveaxis(loc, 0, -1)
        variances_aligned = jnp.moveaxis(variances, 0, -1)

        # 6. Build the N-Dimensional Dense Covariance Matrix
        batched_diag_variances = variances_aligned[..., None, :] * jnp.eye(nfreq)
        dense_cov_matrix = gp_covariances + batched_diag_variances

        # 7. Create Dense Base Distribution
        dense_base = MultivariateNormalFullCovariance(
            loc=loc_aligned, 
            covariance_matrix=dense_cov_matrix
        )

        # 8. Re-apply Bijectors
        # Move the GP event axis (-1) back to the frequency axis (0), THEN apply original bijectors
        bijector_chain = bijectors + [MoveAxis(axis_from=-1, axis_to=0)]
        
        full_bijector = bij.Chain(bijector_chain)
        return dist.Transformed(dense_base, full_bijector)