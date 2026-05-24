import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.simulate.base import AbstractScatteringTerminator, AbstractTransferTerminator, ScatteringResult

class LinearFractionalTerminator(AbstractScatteringTerminator):
    """
    Evaluates the Lower Linear Fractional Transformation (Lower LFT) of a 
    2N-port S-parameter matrix terminated by an N-port S-parameter matrix.
    
    Automatically renormalizes the terminating network if port impedances mismatch.
    """
    s_def: str = eqx.field(default="power", static=True)

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
            
            if self.s_def == "power":
                g = (z_new - z_old) / (z_new + jnp.conj(z_old))
            else:
                g = (z_new - z_old) / (z_new + z_old)
                
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