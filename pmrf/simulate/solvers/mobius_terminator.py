"""pmrf/simulate/terminators.py"""

import jax.numpy as jnp

from pmrf.simulate.base import AbstractTransferTerminator, ScatteringResult

class MobiusTerminator(AbstractTransferTerminator):
    """
    Terminates a two-port ABCD matrix into a one-port S-parameter matrix
    using a Möbius (bilinear) transformation.
    """
    def run(
        self, 
        a_from: jnp.ndarray,  # Shape: (2, 2)
        s_into: jnp.ndarray,  # Shape: (1, 1)
        z0_into: jnp.ndarray, # Shape: (1,)
    ) -> ScatteringResult:
        
        # Terminated load reflection coefficient
        s11 = s_into[0, 0]
        
        A, B = a_from[0, 0], a_from[0, 1]
        C, D = a_from[1, 0], a_from[1, 1]
        
        z0_val = z0_into[0]
        
        num = z0_val * (1 + s11) * (A - z0_val * C) + (B - D * z0_val) * (1 - s11)
        den = z0_val * (1 + s11) * (A + z0_val * C) + (B + D * z0_val) * (1 - s11)
        s11_out = num / den        
        
        # Wrap result back into a (1, 1) matrix
        s_out = jnp.array([[s11_out]])
        
        return ScatteringResult(s=s_out, z0=z0_into)