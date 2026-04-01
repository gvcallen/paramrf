"""
Common likelihood functions for Bayesian inference.
"""

import jax.numpy as jnp
import distreqx.distributions as dist
from typing import Callable, Any

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
    
def distribution_log_likelihood(y_true: jnp.ndarray, y_pred: jnp.ndarray, distribution_fn: Callable[..., Any], **params) -> jnp.ndarray:
    """
    Compute the total log-likelihood of observed data given a general distribution.

    This function instantiates a distribution using the provided `dist_fn`, 
    parameterized by the predictions (`y_pred`) and any additional `params`, 
    and then calculates the sum of the log-probabilities of the true data (`y_true`).

    Parameters
    ----------
    y_true : jnp.ndarray
        The ground truth or measured data.
    y_pred : jnp.ndarray
        The predicted data, typically passed as the primary parameter 
        (e.g., the mean or location) to the distribution function.
    dist_fn : Callable
        A factory function or class constructor that returns a distribution 
        object (e.g., from `distreqx` or `numpyro`). It must accept `y_pred` 
        as its first positional argument alongside the provided `**params`, 
        and the returned object must have a `.log_prob()` method.
    **params : dict
        Additional keyword arguments to parameterize the distribution 
        (e.g., `scale`, `concentration`, `df`).

    Returns
    -------
    jnp.ndarray
        A scalar JAX array representing the sum of the log-likelihoods 
        over all elements in the array.
    """
    return jnp.sum(distribution_fn(y_pred, **params).log_prob(y_true))

def symmetric_gaussian_log_likelihood(y_true: jnp.ndarray, y_pred: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the log-likelihood of a complex-valued function under a 
    symmetric Gaussian assumption.

    Parameters
    ----------
    y_true : jax.Array
        Complex-valued JAX array of shape (nfreq, nports, nports) representing 
        the measured or true data.
    y_pred : jax.Array
        Complex-valued JAX array of shape (nfreq, nports, nports) representing 
        the model prediction.
    sigma : scalar or array_like
        Noise standard deviation. Represents the complex standard deviation.
        Can be a scalar, 2-element array (reflection/transmission), or full (nports**2) array.

    Returns
    -------
    jax.Array
        Scalar JAX array representing the total sum of the log-likelihoods.
    """
    nfreq, nports, _ = y_true.shape
    
    # Broadcast sigma using the shared helper
    sigma_matrix = _broadcast_sigma(sigma, nports)
    
    # Split variance evenly across real and imaginary components
    sigma_parts = sigma_matrix / jnp.sqrt(2.0)
    
    dist_real = dist.Normal(loc=jnp.real(y_pred), scale=sigma_parts)
    dist_imag = dist.Normal(loc=jnp.imag(y_pred), scale=sigma_parts)
    
    log_prob_real = dist_real.log_prob(jnp.real(y_true))
    log_prob_imag = dist_imag.log_prob(jnp.imag(y_true))
    
    return jnp.sum(log_prob_real + log_prob_imag)


def mag_phase_gaussian_log_likelihood(y_true: jnp.ndarray, y_pred: jnp.ndarray, sigma_mag: jnp.ndarray, sigma_phase: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the log-likelihood of a complex-valued function by modeling
    the relative magnitude error and the wrapped phase error as independent Gaussians.

    Parameters
    ----------
    y_true : jax.Array
        Complex-valued JAX array of shape (nfreq, nports, nports) representing 
        the measured or true data.
    y_pred : jax.Array
        Complex-valued JAX array of shape (nfreq, nports, nports) representing 
        the model prediction.
    sigma_mag : scalar or array_like
        Standard deviation for the relative magnitude error (log-magnitude). 
        Can be a scalar, 2-element array (reflection/transmission), or full 
        (nports**2) array.
    sigma_phase : scalar or array_like
        Standard deviation for the phase error (in radians). Can be a scalar, 
        2-element array (reflection/transmission), or full (nports**2) array.

    Returns
    -------
    jax.Array
        Scalar JAX array representing the total sum of the log-likelihoods.
    """
    nfreq, nports, _ = y_true.shape
    
    # Broadcast both sigmas
    sigma_mag_matrix = _broadcast_sigma(sigma_mag, nports)
    sigma_phase_matrix = _broadcast_sigma(sigma_phase, nports)
    
    # --- 1. Relative Magnitude Likelihood ---
    # Log-magnitude naturally models relative error. 
    # Added a small epsilon to prevent log(0) for perfectly zero magnitudes.
    eps = 1e-12
    log_mag_true = jnp.log(jnp.abs(y_true) + eps)
    log_mag_pred = jnp.log(jnp.abs(y_pred) + eps)
    
    dist_mag = dist.Normal(loc=log_mag_pred, scale=sigma_mag_matrix)
    log_prob_mag = dist_mag.log_prob(log_mag_true)
    
    # --- 2. Phase Likelihood ---
    phase_true = jnp.angle(y_true)
    phase_pred = jnp.angle(y_pred)
    
    # Calculate difference and wrap to the [-pi, pi] interval
    phase_diff = phase_true - phase_pred
    wrapped_phase_diff = (phase_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # Model the wrapped error as a zero-mean Gaussian
    dist_phase = dist.Normal(loc=jnp.zeros_like(wrapped_phase_diff), scale=sigma_phase_matrix)
    log_prob_phase = dist_phase.log_prob(wrapped_phase_diff)
    
    return jnp.sum(log_prob_mag + log_prob_phase)

def radial_tangential_gaussian_log_likelihood(y_true: jnp.ndarray, y_pred: jnp.ndarray, sigma_complex: jnp.ndarray, sigma_mag: jnp.ndarray, sigma_phase: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the log-likelihood of a complex-valued function using a locally 
    rotated 2D Gaussian. 
    
    This geometrically aligns the uncertainty distribution with the predicted signal, 
    allowing independent variance scaling along the radial (magnitude) and 
    tangential (phase) axes. This prevents mathematical double-counting of errors 
    while natively modeling the "banana-shaped" uncertainty profile common in RF measurements.

    Parameters
    ----------
    y_true : jax.Array
        Complex-valued JAX array of shape (nfreq, nports, nports) representing 
        measured data.
    y_pred : jax.Array
        Complex-valued JAX array of shape (nfreq, nports, nports) representing 
        the model prediction.
    sigma_complex : scalar or array_like
        Standard deviation for the absolute symmetric complex noise floor 
        (e.g., thermal/trace noise). Can be a scalar, 2-element array 
        (reflection/transmission), or full (nports**2) array.
    sigma_mag : scalar or array_like
        Standard deviation for the relative magnitude error (fractional). 
        Can be a scalar, 2-element array (reflection/transmission), or 
        full (nports**2) array.
    sigma_phase : scalar or array_like
        Standard deviation for the absolute phase error (in radians). 
        Can be a scalar, 2-element array (reflection/transmission), or 
        full (nports**2) array.

    Returns
    -------
    jax.Array
        Scalar JAX array representing the total sum of the positive log-likelihoods.
    """
    nfreq, nports, _ = y_true.shape
    
    # Broadcast all three sigmas into (nports, nports) matrices
    sigma_complex_mat = _broadcast_sigma(sigma_complex, nports)
    sigma_mag_mat = _broadcast_sigma(sigma_mag, nports)
    sigma_phase_mat = _broadcast_sigma(sigma_phase, nports)
    
    # 1. Extract Magnitudes and Phases of the predicted signal
    mag_pred = jnp.abs(y_pred)
    phase_pred = jnp.angle(y_pred)
    
    # 2. Compute the complex error and rotate it to the local frame of y_pred
    # Multiplying by exp(-i * phase) rotates the error so the Real axis aligns 
    # radially (magnitude error) and the Imaginary axis aligns tangentially (phase error).
    error = y_true - y_pred
    error_rot = error * jnp.exp(-1j * phase_pred)
    
    # Extract the independent radial and tangential error components
    err_radial = jnp.real(error_rot)
    err_tangential = jnp.imag(error_rot)
    
    # 3. Calculate the combined variances for the radial and tangential axes.
    # The symmetric VNA noise is split evenly between both axes (variance / 2).
    # The magnitude and phase variances scale linearly with the signal's magnitude.
    var_vna_per_axis = (sigma_complex_mat ** 2) / 2.0
    
    var_radial = var_vna_per_axis + (mag_pred * sigma_mag_mat) ** 2
    var_tangential = var_vna_per_axis + (mag_pred * sigma_phase_mat) ** 2
    
    # Convert variances back to standard deviations (scale) for the normal distributions
    scale_radial = jnp.sqrt(var_radial)
    scale_tangential = jnp.sqrt(var_tangential)
    
    # 4. Evaluate the zero-mean distributions for both axes
    dist_radial = dist.Normal(loc=jnp.zeros_like(err_radial), scale=scale_radial)
    dist_tangential = dist.Normal(loc=jnp.zeros_like(err_tangential), scale=scale_tangential)
    
    # Calculate log probabilities
    log_prob_radial = dist_radial.log_prob(err_radial)
    log_prob_tangential = dist_tangential.log_prob(err_tangential)
    
    # Return the sum of the log-likelihoods over all frequencies and port interactions
    return jnp.sum(log_prob_radial + log_prob_tangential)