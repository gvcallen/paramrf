"""pmrf/simulate/redheffer.py"""

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.math import nudge_diag
from pmrf.simulate.base import AbstractScatteringCascader, ScatteringResult


class Redheffer(AbstractScatteringCascader):
    """Cascades a stacked sequence of S-parameter networks using Redheffer's star product."""
    eps: float = eqx.field(default=1e-12, static=True)
    
    def run(
        self, 
        s_stacked: jnp.ndarray, # Shape: (N_networks, N_ports, N_ports)
        port_z0: jnp.ndarray,   # Shape: (N_networks, N_ports)
    ) -> ScatteringResult:
        
        # Fast path for single network
        if s_stacked.shape[0] == 1:
            return ScatteringResult(s=s_stacked[0], z0=port_z0[0])

        # jax.lax.scan loops over the component arrays dynamically without unrolling
        def scan_fn(carry, x):
            S_acc, z0_acc = carry
            S_i, z0_i = x
            S_next, z0_next = self.cascade_two(S_acc, z0_acc, S_i, z0_i)
            return (S_next, z0_next), None

        (S_cas, z0_cas), _ = jax.lax.scan(
            scan_fn, 
            init=(s_stacked[0], port_z0[0]), 
            xs=(s_stacked[1:], port_z0[1:])
        )
        
        return ScatteringResult(s=S_cas, z0=z0_cas)
        
    def cascade_two(
        self,
        Smat_A: jnp.ndarray, # Shape: (N_ports, N_ports)
        z0_A: jnp.ndarray,   # Shape: (N_ports,)
        Smat_B: jnp.ndarray,
        z0_B: jnp.ndarray,
    ):
        nports = Smat_A.shape[0]
        N = nports // 2

        # Concatenate outer ports: Left ports of A and Right ports of B
        z0_cas = jnp.concatenate((z0_A[:N], z0_B[N:]), axis=0)

        # Partition blocks
        A11 = Smat_A[:N, :N]
        A12 = Smat_A[:N, N:]
        A21 = Smat_A[N:, :N]
        A22 = Smat_A[N:, N:]

        B11 = Smat_B[:N, :N]
        B12 = Smat_B[:N, N:]
        B21 = Smat_B[N:, :N]
        B22 = Smat_B[N:, N:]

        I = jnp.eye(N, dtype=Smat_A.dtype)

        M = nudge_diag(I - B11 @ A22, eps=self.eps)
        N_mat = nudge_diag(I - A22 @ B11, eps=self.eps) # Renamed from N
        X = jnp.linalg.solve(M, I)
        Y = jnp.linalg.solve(N_mat, I)

        S11 = A11 + A12 @ X @ B11 @ A21
        S12 = A12 @ X @ B12
        S21 = B21 @ Y @ A21
        S22 = B22 + B21 @ Y @ A22 @ B12

        # Recombine blocks into a single matrix
        top = jnp.concatenate((S11, S12), axis=1)
        bottom = jnp.concatenate((S21, S22), axis=1)
        S_cas = jnp.concatenate((top, bottom), axis=0)

        return S_cas, z0_cas