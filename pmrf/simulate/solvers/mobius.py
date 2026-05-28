"""pmrf/simulate/terminators.py"""

import jax.numpy as jnp

from pmrf.simulate.base import AbstractTransferTerminator, ScatteringResult
from pmrf.utils import error_if

class MobiusTerminator(AbstractTransferTerminator):
    def run(
        self, 
        a_from: jnp.ndarray,  # Shape: (2, 2)
        s_into: jnp.ndarray,  # Shape: (1, 1)
        z0_into: jnp.ndarray, # Shape: (1,)
    ) -> ScatteringResult:
        
        # Enforce shape constraints
        a_from = error_if(a_from, a_from.shape != (2, 2), f"MobiusTerminator requires a_from shape (2, 2), got {a_from.shape}")
        s_into = error_if(s_into, s_into.shape != (1, 1), f"MobiusTerminator requires s_into shape (1, 1), got {s_into.shape}")
        z0_into = error_if(z0_into, z0_into.shape != (1,), f"MobiusTerminator requires z0_into shape (1,), got {z0_into.shape}")

        # Terminated load reflection coefficient
        s11 = s_into[0, 0]
        
        A, B = a_from[0, 0], a_from[0, 1]
        C, D = a_from[1, 0], a_from[1, 1]
        
        z0_val = z0_into[0]
        
        num = z0_val * (1 + s11) * (A - z0_val * C) + (B - D * z0_val) * (1 - s11)
        den = z0_val * (1 + s11) * (A + z0_val * C) + (B + D * z0_val) * (1 - s11)
        
        # Prevent silent division-by-zero NaNs during runtime
        den = error_if(den, jnp.abs(den) == 0.0, "Singular matrix encountered in Möbius transformation (division by zero).")

        s11_out = num / den        
        
        # Wrap result back into a (1, 1) matrix
        s_out = jnp.array([[s11_out]])
        
        return ScatteringResult(s=s_out, z0=z0_into)