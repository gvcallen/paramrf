"""
Models that represent the noise of a measurements, generally used in a likelihood.
"""
import jax.numpy as jnp
from typing import Callable

import parax as prx
from pmrf.core import NoiseModel


class ReflectionTransmissionNoise(NoiseModel):
    """
    Reflection and transmission coefficient noise model.
    
    Maps underlying `gamma` and `tau` noises
    to a full matrix based on the specified port axes.
    
    Supports both circularly-symmetric underlying noise (returns a single array) 
    and general complex noise (returns a tuple of (hermitian, pseudo)).
    """    
    gamma: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]]
    tau: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]]
    
    port_axes: tuple[int, int] = prx.field(static=True, default=(-2, -1))
    
    def _build_matrix(self, val_gamma: jnp.ndarray, val_tau: jnp.ndarray, target_shape: tuple) -> jnp.ndarray:
        """Helper to map diagonal/off-diagonal values using robust broadcasting."""
        # 1. Extract nports dynamically from the target shape
        ax1 = self.port_axes[0] % len(target_shape)
        ax2 = self.port_axes[1] % len(target_shape)
        
        nports = target_shape[ax1]
        if nports != target_shape[ax2]:
            raise ValueError(
                f"Dimensions at port_axes {self.port_axes} must be equal for a square matrix. "
                f"Got {target_shape[ax1]} and {target_shape[ax2]}."
            )
            
        # 2. Create a base boolean identity matrix for the ports
        eye = jnp.eye(nports, dtype=bool)
        
        # 3. Reshape the identity matrix so it broadcasts correctly with target_shape
        eye_shape = [1] * len(target_shape)
        eye_shape[ax1] = nports
        eye_shape[ax2] = nports
        
        eye_broadcastable = eye.reshape(eye_shape)
        
        # 4. Use jnp.where to conditionally select gamma or tau
        return jnp.where(eye_broadcastable, val_gamma, val_tau)

    def __call__(self, y_pred: jnp.ndarray):
        val_gamma = self.gamma(y_pred) if callable(self.gamma) else self.gamma
        val_tau = self.tau(y_pred) if callable(self.tau) else self.tau
        
        is_gamma_tuple = isinstance(val_gamma, tuple)
        is_tau_tuple = isinstance(val_tau, tuple)
        
        if not is_gamma_tuple and not is_tau_tuple:
            # Standard real or circularly-symmetric complex case
            return self._build_matrix(val_gamma, val_tau, y_pred.shape)
            
        else:
            # General complex case: Route both Hermitian and Pseudo variances
            gamma_h = val_gamma[0] if is_gamma_tuple else val_gamma
            gamma_p = val_gamma[1] if is_gamma_tuple else jnp.zeros_like(gamma_h)
            
            tau_h = val_tau[0] if is_tau_tuple else val_tau
            tau_p = val_tau[1] if is_tau_tuple else jnp.zeros_like(tau_h)
            
            matrix_h = self._build_matrix(gamma_h, tau_h, y_pred.shape)
            matrix_p = self._build_matrix(gamma_p, tau_p, y_pred.shape)
            
            return matrix_h, matrix_p


class RadialTangentialNoise(NoiseModel):
    """
    Radial/tangential complex-valued heteroscedastic variance noise model.
    
    Models noise as relative radial and tangential variance that scales
    with the squared magnitude of the signal. The parameters `magnitude` 
    and `phase` represent the relative variance components.
    
    Returns the hermitian and pseudo variance as a tuple.
    """
    magnitude: prx.Parameter | jnp.ndarray | Callable
    phase: prx.Parameter | jnp.ndarray | Callable

    def __call__(self, y_pred: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Unpack callables
        var_rad_rel = self.magnitude(y_pred) if callable(self.magnitude) else self.magnitude
        var_tan_rel = self.phase(y_pred) if callable(self.phase) else self.phase
        
        # 1. Total Hermitian Variance (Gamma)
        # Gamma = |y|^2 * (V_rad + V_tan)
        mag_sq = jnp.real(y_pred * jnp.conj(y_pred))
        variance = mag_sq * (var_rad_rel + var_tan_rel)
        
        # 2. Pseudo-Variance (C)
        # C = y^2 * (V_rad - V_tan)
        pseudo_covariance = (y_pred**2) * (var_rad_rel - var_tan_rel)
        
        return variance, pseudo_covariance