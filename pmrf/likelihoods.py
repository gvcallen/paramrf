"""
Stateful likelihood modules for Bayesian inference and probabilistic modeling.

These classes wrap pure mathematical log-likelihood functions into a :class:``pmrf.Metric``.

All likelihoods take the true and predicted arrays as inputs, and return the log-likelihood.
It is assume that the underlying distribution is centred around the predicted value, meaning
the likelihood returns the log probability of observed the true value given that prediction.
"""

import jax.numpy as jnp
import parax as prx

from pmrf.math import likelihoods as F
from pmrf.core import Metric

def _add_sigmas(a, b):
    return jnp.sqrt(jnp.square(a) + jnp.square(b))

class GaussianLikelihood(Metric):
    """
    Gaussian log-likelihood metric.

    Attributes
    ----------
    sigma : float | jnp.ndarray | prx.Parameter
        Noise standard deviation. Can be a scalar, 2-element array (S11/S21), 
        or a full nports**2 array.
    """
    sigma: float | jnp.ndarray | prx.Parameter

    def __call__(
        self, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray,
        sigma_delta: jnp.ndarray = 0.0,
    ) -> jnp.ndarray:
        sigma_total = _add_sigmas(self.sigma, sigma_delta)
        
        return F.gaussian_log_likelihood(
            y_true=y_true, 
            y_pred=y_pred, 
            sigma=sigma_total,
        )

class SymmetricGaussianLikelihood(Metric):
    """
    Circularly symmetric, complex Gaussian log-likelihood metric.

    Models the complex error with equal variance across the real and imaginary axes.

    Attributes
    ----------
    sigma : float | jnp.ndarray | prx.Parameter
        Noise standard deviation. Can be a scalar, 2-element array (S11/S21), 
        or a full nports**2 array.
    """
    sigma: float | jnp.ndarray | prx.Parameter

    def __call__(
        self, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray,
        sigma_delta: jnp.ndarray = 0.0,
    ) -> jnp.ndarray:
        sigma_total = _add_sigmas(self.sigma, sigma_delta)
        
        return F.symmetric_gaussian_log_likelihood(
            y_true=y_true, 
            y_pred=y_pred, 
            sigma=sigma_total,
        )


class MagnitudePhaseGaussianLikelihood(Metric):
    """
    Magnitude/Phase Gaussian log-likelihood metric.

    Models the relative magnitude error and wrapped phase error as independent Gaussians.

    Attributes
    ----------
    sigma_mag : float | jnp.ndarray | prx.Parameter
        Standard deviation for the relative magnitude error (log-magnitude).
    sigma_phase : float | jnp.ndarray | prx.Parameter
        Standard deviation for the phase error (in radians).
    """
    sigma_mag: float | jnp.ndarray | prx.Parameter
    sigma_phase: float | jnp.ndarray | prx.Parameter

    def __call__(
        self, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray,
        sigma_mag_delta: jnp.ndarray = 0.0,
        sigma_phase_delta: jnp.ndarray = 0.0,
    ) -> jnp.ndarray:
        sigma_mag_total = _add_sigmas(self.sigma_mag, sigma_mag_delta)
        sigma_phase_total = _add_sigmas(self.sigma_phase, sigma_phase_delta)
        
        return F.magnitude_phase_gaussian_log_likelihood(
            y_true=y_true, 
            y_pred=y_pred, 
            sigma_mag=sigma_mag_total, 
            sigma_phase=sigma_phase_total,
        )


class RadialTangentialGaussianLikelihood(Metric):
    """
    Radial-Tangential Gaussian log-likelihood metric.

    Geometrically aligns the uncertainty distribution with the predicted signal 
    to natively model the "banana-shaped" uncertainty profile of RF measurements.

    Attributes
    ----------
    sigma_complex : float | jnp.ndarray | prx.Parameter
        Absolute symmetric complex noise floor (e.g., thermal noise).
    sigma_mag : float | jnp.ndarray | prx.Parameter
        Relative magnitude error standard deviation.
    sigma_phase : float | jnp.ndarray | prx.Parameter
        Absolute phase error standard deviation (radians).
    """
    sigma_complex: float | jnp.ndarray | prx.Parameter
    sigma_mag: float | jnp.ndarray | prx.Parameter
    sigma_phase: float | jnp.ndarray | prx.Parameter

    def __call__(
        self, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray,
        sigma_complex_delta: jnp.ndarray = 0.0,
        sigma_mag_delta: jnp.ndarray = 0.0,
        sigma_phase_delta: jnp.ndarray = 0.0,
    ) -> jnp.ndarray:
        sigma_complex_total = _add_sigmas(self.sigma_complex, sigma_complex_delta)        
        sigma_mag_total = _add_sigmas(self.sigma_mag, sigma_mag_delta)        
        sigma_phase_total = _add_sigmas(self.sigma_phase, sigma_phase_delta)        
        
        return F.radial_tangential_gaussian_log_likelihood(
            y_true=y_true,
            y_pred=y_pred, 
            sigma_complex=sigma_complex_total, 
            sigma_mag=sigma_mag_total, 
            sigma_phase=sigma_phase_total,
        )

__all__ = [
    'DistributionLikelihood',
    'SymmetricGaussianLikelihood',
    'MagnitudePhaseGaussianLikelihood',
    'RadialTangentialGaussianLikelihood',
    'likelihood_from_alias'
]