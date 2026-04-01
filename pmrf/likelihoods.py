"""
Stateful likelihood modules for Bayesian inference and probabilistic modeling.

These classes wrap pure mathematical log-likelihood functions into a :class:``pmrf.Metric``.
"""

import jax.numpy as jnp
import parax as prx

from pmrf.math import likelihoods as F
from pmrf.core import Metric

class SymmetricGaussianLikelihood(Metric):
    """
    Symmetric Gaussian log-likelihood metric.

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
    ) -> jnp.ndarray:
        return F.symmetric_gaussian_log_likelihood(
            y_true=y_true, 
            y_pred=y_pred, 
            sigma=self.sigma
        )


class MagPhaseGaussianLikelihood(Metric):
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
    ) -> jnp.ndarray:
        return F.mag_phase_gaussian_log_likelihood(
            y_true=y_true, 
            y_pred=y_pred, 
            sigma_mag=self.sigma_mag, 
            sigma_phase=self.sigma_phase
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
    ) -> jnp.ndarray:
        return F.radial_tangential_gaussian_log_likelihood(
            y_true=y_true, 
            y_pred=y_pred, 
            sigma_complex=self.sigma_complex, 
            sigma_mag=self.sigma_mag, 
            sigma_phase=self.sigma_phase
        )

__all__ = [
    'DistributionLikelihood',
    'SymmetricGaussianLikelihood',
    'MagPhaseGaussianLikelihood',
    'RadialTangentialGaussianLikelihood',
    'likelihood_from_alias'
]