"""
Composite models that terminate a network into a load network.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Literal

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import field
from pmrf.types import ArrayLike
from pmrf.rf import s2a

EVAL_Z0 = 50.0

class Terminated(Model):
    """
    Represents one network terminated in another.

    Mathematically collapses an active network by terminating a subset 
    of its ports with a known load matrix.

    Parameters
    ----------
    terminated_from : Model
        The model being terminated.
    terminated_into : Model
        The load model that `terminated_from` is terminated into.
    method : {'s', 'a'}, default='s'
        The underlying mathematical domain to use for the termination reduction.
    """
    #: The "from" model.
    terminated_from: Model

    #: The "into" model.
    terminated_into: Model
    
    #: The termination reduction algorithm method.
    method: Literal['s', 'a'] = field(default='s', kw_only=True)

    def __post_init__(self):
        if self.terminated_from.nports != 2 * self.terminated_into.nports:
            raise ValueError("Terminated only supports terminating 2N port networks in a 1N port")

    @property
    def number_of_ports(self):
        return self.terminated_from.number_of_ports - self.terminated_into.number_of_ports

    def expand(self):
        P = self.terminated_from.nports
        M = self.terminated_into.nports
        K = P - M

        port_mapping = [(self.terminated_from, i) for i in range(K)]

        internal_connections = [
            [(self.terminated_from, K + i), (self.terminated_into, i)]
            for i in range(M)
        ]

        return port_mapping, internal_connections        

    # --- TERMINATION ALGORITHMS (Single Frequency Point) ---

    def _terminate_two_s(
        self, 
        s_from: jnp.ndarray,
        z0_from: jnp.ndarray,
        s_into: jnp.ndarray,
        z0_into: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Exact boundary condition substitution for S-parameters."""
        P = s_from.shape[0]      # Total ports in the original matrix
        M = s_into.shape[0]      # Ports being terminated
        K = P - M                # Surviving ports
        
        S11 = s_from[:K, :K]
        S12 = s_from[:K, K:]
        S21 = s_from[K:, :K]
        S22 = s_from[K:, K:]
        
        z0_out = z0_from[K:]
        
        def apply_renorm(operand):
            S_L, z_old, z_new = operand
            
            g = (z_new - z_old) / (z_new + jnp.conj(z_old))
            G = jnp.diag(g)
            I = jnp.eye(M, dtype=S_L.dtype)
            
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

        I = jnp.eye(M, dtype=s_from.dtype)
        diff = I - S22 @ S_L_matched
        X = jnp.linalg.solve(diff, S21)
        
        S_term = S11 + S12 @ S_L_matched @ X
        z0_term = z0_from[:K]
        
        return S_term, z0_term

    def _terminate_two_abcd(
        self, 
        a_from: jnp.ndarray,  # Shape: (2, 2)
        s_into: jnp.ndarray,  # Shape: (1, 1)
        z0_into: jnp.ndarray, # Shape: (1,)
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Exact Möbius transformation for terminating ABCD into S-parameters."""
        # Enforce shape constraints
        a_from = eqx.error_if(a_from, a_from.shape != (2, 2), f"ABCD termination requires a_from shape (2, 2), got {a_from.shape}")
        s_into = eqx.error_if(s_into, s_into.shape != (1, 1), f"ABCD termination requires s_into shape (1, 1), got {s_into.shape}")
        z0_into = eqx.error_if(z0_into, z0_into.shape != (1,), f"ABCD termination requires z0_into shape (1,), got {z0_into.shape}")

        # Terminated load reflection coefficient
        s11 = s_into[0, 0]
        
        A, B = a_from[0, 0], a_from[0, 1]
        C, D = a_from[1, 0], a_from[1, 1]
        
        z0_val = z0_into[0]
        
        num = z0_val * (1 + s11) * (A - z0_val * C) + (B - D * z0_val) * (1 - s11)
        den = z0_val * (1 + s11) * (A + z0_val * C) + (B + D * z0_val) * (1 - s11)
        
        # Prevent silent division-by-zero NaNs during runtime
        den = eqx.error_if(den, jnp.abs(den) == 0.0, "Singular matrix encountered in Möbius transformation (division by zero).")

        s11_out = num / den        
        
        # Wrap result back into a (1, 1) matrix
        s_out = jnp.array([[s11_out]])
        
        return s_out, z0_into

    # --- SIMULATION & CONVERSION ---

    def _solve(self, freq: Frequency, z0: ArrayLike = EVAL_Z0) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Dispatches data prep and solving across the active vmapped mathematical method."""
        # The load is always required as an S-parameter matrix (reflection)
        s_into = self.terminated_into.s(freq, z0=z0)
        
        # Broadcast port impedance for the load
        z0_dtype = s_into.dtype
        M = self.terminated_into.nports
        Nf = freq.npoints
        z0_into = jnp.broadcast_to(jnp.asarray(z0, dtype=z0_dtype), (Nf, M))

        if self.method == 's':
            s_from = self.terminated_from.s(freq, z0=z0)
            P = self.terminated_from.nports
            z0_from = jnp.broadcast_to(jnp.asarray(z0, dtype=z0_dtype), (Nf, P))
            
            run_vmap = jax.vmap(self._terminate_two_s, in_axes=(0, 0, 0, 0))
            return run_vmap(s_from, z0_from, s_into, z0_into)
            
        elif self.method == 'a':
            a_from = self.terminated_from.a(freq)
            
            run_vmap = jax.vmap(self._terminate_two_abcd, in_axes=(0, 0, 0))
            return run_vmap(a_from, s_into, z0_into)
            
        else:
            raise ValueError(f"Unknown termination method: {self.method}")

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s_term, _ = self._solve(freq, z0=z0)
        return s_term

    def a(self, freq: Frequency) -> jnp.ndarray:
        s_term, z0_term = self._solve(freq, z0=EVAL_Z0)
        return s2a(s_term, z0=z0_term)