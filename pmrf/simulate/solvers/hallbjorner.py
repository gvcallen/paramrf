"""pmrf/simulate/hallbjorner.py"""

import jax
import jax.numpy as jnp

from pmrf.simulate.base import AbstractScatteringReducer, PortRepresentation, ScatteringResult
from pmrf.rf.conversions import s2s

class Hallbjorner(AbstractScatteringReducer):
    """
    Standard S-parameter solver using Hallbjorner's method.
    Re-written purely in vector math. No for-loops, no slice-packing.
    """
    s_layout: str = 'block_diagonal'

    def run(
        self, 
        s_matrices: jax.Array,  # Shape: (num_ports, num_ports)
        port_z0: jax.Array,           # Shape: (num_ports,)
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        S = s_matrices
        z0 = port_z0
        
        S_trav = s2s(S, z0, 'traveling', 'power')
        
        mask = jnp.array(
            topology.port_to_net_map[:, None] == topology.port_to_net_map[None, :], 
            dtype=S.dtype
        )
        
        # Calculate X directly over the full N x N matrix space
        y0 = 1.0 / z0
        y0_sum = jnp.dot(mask, y0) # Sums y0 for all ports sharing the same net
        y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
        
        # Process ALL connections at once
        X = mask * ((2.0 * y0_sqrt_prod) / y0_sum[:, None] - jnp.eye(topology.num_ports, dtype=S.dtype))
        
        # Compute Intermediate matrix [T]
        # Because X aligns perfectly with the standard block-diagonal S matrix, 
        # we don't need to permute/reorder S into a C matrix.
        T = jnp.eye(topology.num_ports, dtype=S.dtype) - S_trav @ X
        
        # Extract Sub-matrices
        ext = topology.ext_idx
        int_ = topology.int_idx
        
        T_A = T[jnp.ix_(ext, ext)]
        T_B = T[jnp.ix_(ext, int_)]
        T_C = T[jnp.ix_(int_, ext)]
        T_D = T[jnp.ix_(int_, int_)]
        
        X_A = X[jnp.ix_(ext, ext)]
        X_B = X[jnp.ix_(ext, int_)]
        
        # Schur Complement Reduction
        # Using pinv as in your original code for stability on perfect isolations
        tmp_mat = jnp.linalg.pinv(T_D) @ T_C 
        
        numerator = X_A - X_B @ tmp_mat
        denominator = T_A - T_B @ tmp_mat
        S_ext_trav = numerator @ jnp.linalg.inv(denominator)
        
        # Convert back to power waves
        z0_ext = z0[ext]
        S_ext_power = s2s(S_ext_trav, z0_ext, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_ext)