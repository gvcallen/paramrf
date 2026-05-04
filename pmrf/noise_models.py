"""
Models that represent the noise of a measurement, used within in a likelihood.
"""
import jax.numpy as jnp

import parax as prx
from pmrf.core import NoiseModel


class AutoCrossNoise(NoiseModel):
    """
    Auto and cross term noise model.
    
    Maps underlying `auto` and `cross` noise models
    to a full matrix based on the specified port axes.

    Can be used to assign separate noise variances
    to reflection and (auto) transmission (cross) coefficients.

    Operates in "event space". For example, for a standard
    N-port S-parameter feature, the input `y_event` will be
    of shape (nports, nports, nfreq) or (nports, nports, 2, nfreq).
    """    
    auto: prx.Param
    cross: prx.Param
    
    port_axes: tuple[int, int] = prx.constrained(static=True, default=(0, 1))
    
    def __call__(self, y_event: jnp.ndarray):
        val_gamma, val_tau = self.auto, self.cross
        target_shape = y_event.shape
        
        ax1 = self.port_axes[0] % len(target_shape)
        ax2 = self.port_axes[1] % len(target_shape)
        
        nports = target_shape[ax1]
        if nports != target_shape[ax2]:
            raise ValueError(
                f"Dimensions at port_axes {self.port_axes} must be equal for a square matrix. "
                f"Got {target_shape[ax1]} and {target_shape[ax2]}."
            )
            
        eye = jnp.eye(nports, dtype=bool)
        eye_shape = [1] * len(target_shape)
        eye_shape[ax1] = nports
        eye_shape[ax2] = nports
        eye_broadcastable = eye.reshape(eye_shape)
        
        return jnp.where(eye_broadcastable, val_gamma, val_tau)    

# class RadialTangentialNoise(NoiseModel):
#     """
#     Radial/tangential complex-valued heteroscedastic variance noise model.
    
#     Models noise as relative radial and tangential variance that scales
#     with the squared magnitude of the signal. The parameters `magnitude` 
#     and `phase` represent the relative variance components.
    
#     Returns the hermitian and pseudo variance as a tuple.
#     """
#     magnitude: prx.Parameter | jnp.ndarray | Callable
#     phase: prx.Parameter | jnp.ndarray | Callable

#     def __call__(self, y_pred: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
#         # Unpack callables
#         var_rad_rel = self.magnitude(y_pred) if callable(self.magnitude) else self.magnitude
#         var_tan_rel = self.phase(y_pred) if callable(self.phase) else self.phase
        
#         # 1. Total Hermitian Variance (Gamma)
#         # Gamma = |y|^2 * (V_rad + V_tan)
#         mag_sq = jnp.real(y_pred * jnp.conj(y_pred))
#         variance = mag_sq * (var_rad_rel + var_tan_rel)
        
#         # 2. Pseudo-Variance (C)
#         # C = y^2 * (V_rad - V_tan)
#         pseudo_covariance = (y_pred**2) * (var_rad_rel - var_tan_rel)
        
#         return variance, pseudo_covariance