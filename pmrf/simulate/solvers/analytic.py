import jax
import jax.numpy as jnp

from pmrf.simulate.base import AbstractABCDTerminator, AbstractABCDCascader, AbstractScatteringTerminator, ScatteringResult, TransferResult
from pmrf.utils import error_if

class BilinearABCDTerminator(AbstractABCDTerminator):
    def run(
        self, 
        a_from: jnp.ndarray,  # Shape: (2, 2)
        s_into: jnp.ndarray,  # Shape: (1, 1)
        z0_into: jnp.ndarray, # Shape: (1,)
    ) -> ScatteringResult:
        
        # Enforce shape constraints
        a_from = error_if(a_from, a_from.shape != (2, 2), f"AnalyticABCDTerminator requires a_from shape (2, 2), got {a_from.shape}")
        s_into = error_if(s_into, s_into.shape != (1, 1), f"AnalyticABCDTerminator requires s_into shape (1, 1), got {s_into.shape}")
        z0_into = error_if(z0_into, z0_into.shape != (1,), f"AnalyticABCDTerminator requires z0_into shape (1,), got {z0_into.shape}")

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
    

class AnalyticABCDCascader(AbstractABCDCascader):
    def run(
        self, 
        a_stacked: jnp.ndarray, # Shape: (N_networks, N_ports, N_ports)
    ) -> TransferResult:
        if a_stacked.shape[0] == 1:
            return TransferResult(a=a_stacked[0])

        def scan_fn(carry, x):
            return carry @ x, None

        a_cas, _ = jax.lax.scan(
            scan_fn, 
            init=a_stacked[0], 
            xs=a_stacked[1:]
        )
        
        return TransferResult(a=a_cas)
    

import jax
import jax.numpy as jnp

from pmrf.simulate.base import AbstractScatteringTerminator, ScatteringResult

class AnalyticScatteringTerminator(AbstractScatteringTerminator):
    def run(
        self, 
        s_from: jnp.ndarray,  # Shape: (2N, 2N)
        z0_from: jnp.ndarray, # Shape: (2N,)
        s_into: jnp.ndarray,  # Shape: (N, N)
        z0_into: jnp.ndarray, # Shape: (N,)
    ) -> ScatteringResult:
        
        N = s_into.shape[0]
        
        # Extract block matrices
        S11 = s_from[:N, :N]
        S12 = s_from[:N, N:]
        S21 = s_from[N:, :N]
        S22 = s_from[N:, N:]
        
        # Impedances of the connecting ports
        z0_out = z0_from[N:]
        
        # --- Renormalization Block ---
        def apply_renorm(operand):
            S_L, z_old, z_new = operand
            
            g = (z_new - z_old) / (z_new + jnp.conj(z_old))
                
            G = jnp.diag(g)
            I = jnp.eye(N, dtype=S_L.dtype)
            
            I_minus_G = I - G
            
            X = jnp.linalg.solve(I_minus_G, S_L - G)          
            Z = jnp.linalg.solve(I - G @ S_L, I_minus_G)      
            
            return X @ Z

        def skip_renorm(operand):
            S_L, _, _ = operand
            return S_L

        needs_renorm = jnp.logical_not(jnp.allclose(z0_out, z0_into))
        
        S_L_matched = jax.lax.cond(
            needs_renorm,
            apply_renorm,
            skip_renorm,
            (s_into, z0_into, z0_out)
        )
        # -----------------------------

        I = jnp.eye(N, dtype=s_from.dtype)

        # Compute S_term = S11 + S12 @ S_L @ inv(I - S22 @ S_L) @ S21
        diff = I - S22 @ S_L_matched
        X = jnp.linalg.solve(diff, S21)
        
        S_term = S11 + S12 @ S_L_matched @ X
        z0_term = z0_from[:N]
        
        return ScatteringResult(s=S_term, z0=z0_term)