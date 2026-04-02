import jax.numpy as jnp
import parax as prx

from pmrf.core import Likelihood

def _broadcast_sigma(sigma, nports):
    """
    Broadcast a sigma parameter into a full (nports, nports) matrix.
    
    Accepts a scalar, a 2-element array (reflection, transmission), or a 
    full nports**2 array and reshapes/broadcasts it accordingly.

    Parameters
    ----------
    sigma : scalar or array_like
        The uncertainty parameter to broadcast. Can be a scalar, a 2-element 
        array (where index 0 is reflection and index 1 is transmission), 
        or a full array of size nports**2.
    nports : int
        The number of ports, defining the shape of the output matrix.

    Returns
    -------
    jax.Array
        A 2D JAX array of shape (nports, nports) representing the broadcasted 
        sigma matrix.

    Raises
    ------
    ValueError
        If the size of `sigma` is not 1, 2, or nports**2.
    """
    sigma = jnp.asarray(sigma)
    
    if sigma.size == 1:
        # Scalar: Used for all ports
        return jnp.full((nports, nports), sigma.squeeze())
        
    elif sigma.size == 2:
        # Two sigmas: sigma[0] for reflection (diag), sigma[1] for transmission (off-diag)
        sigma_matrix = jnp.full((nports, nports), sigma[1])
        diag_indices = jnp.diag_indices(nports)
        return sigma_matrix.at[diag_indices].set(sigma[0])
        
    elif sigma.size == nports ** 2:
        # Full matrix: represents each port-to-port interaction
        return sigma.reshape((nports, nports))
        
    else:
        raise ValueError(f"Invalid size for sigma: {sigma.size}. Expected 1, 2, or {nports**2}.")
    
from distreqx.distributions import MultivariateNormalDiag

class RadialTangentialObservation(Likelihood):
    """
    Geometrically aligns the uncertainty distribution with the predicted signal, 
    allowing independent variance scaling along the radial and tangential axes.
    """
    sigma_complex: prx.Parameter | float | jnp.ndarray
    sigma_mag: prx.Parameter | float | jnp.ndarray
    sigma_phase: prx.Parameter | float | jnp.ndarray

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        # -----------------------------------------------------------------
        # 1. RESIDUALS (The Geometry)
        # -----------------------------------------------------------------
        phase_pred = jnp.angle(y_pred)
        error_rot = (y_true - y_pred) * jnp.exp(-1j * phase_pred)
        err_radial = jnp.real(error_rot)
        err_tangential = jnp.imag(error_rot)
        residuals = jnp.stack([err_radial, err_tangential], axis=-1)
        mag_pred = jnp.abs(y_pred)
        
        # -----------------------------------------------------------------
        # 2. VARIANCES (The Heteroscedastic Noise)
        # -----------------------------------------------------------------
        nports = y_true.shape[1] if y_true.ndim > 1 else 1
        sig_c = _broadcast_sigma(self.sigma_complex, nports)
        sig_m = _broadcast_sigma(self.sigma_mag, nports)
        sig_p = _broadcast_sigma(self.sigma_phase, nports)
        
        var_vna_per_axis = (sig_c ** 2) / 2.0
        
        var_radial = var_vna_per_axis + (mag_pred * sig_m) ** 2
        var_tangential = var_vna_per_axis + (mag_pred * sig_p) ** 2
        
        # Stack to perfectly mirror the residual shape
        variances = jnp.stack([var_radial, var_tangential], axis=-1)

        return residuals, variances
    
import distreqx.distributions as dist
    
def radial_tangential_noise(err_radial, err_tangential, y_mag, sigma_complex, sigma_mag, sigma_phase):
    var_vna_per_axis = (sigma_complex) / 2.0
    
    var_radial = var_vna_per_axis + (y_mag * sigma_mag) ** 2
    var_tangential = var_vna_per_axis + (y_mag * sigma_phase) ** 2
    
    # Stack to perfectly mirror the residual shape
    variances = jnp.stack([var_radial, var_tangential], axis=-1)
    
    