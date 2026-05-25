"""pmrf/simulate/hallbjorner.py"""

import jax
import jax.numpy as jnp

from pmrf.simulate.base import AbstractScatteringReducer, PortRepresentation, ScatteringResult
from pmrf.rf.conversions import s2s

class Hallbjorner(AbstractScatteringReducer):
    def run(
        self, 
        s_block_diagonal: jax.Array,  # Shape: (num_ports, num_ports)
        z0_ports: jax.Array,           # Shape: (num_ports,)
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        # Convert to traveling waves which the original algorithm uses
        S_trav = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        
        mask = jnp.array(
            topology.port_to_net_map[:, None] == topology.port_to_net_map[None, :], 
            dtype=s_block_diagonal.dtype
        )
        
        # Calculate X and process all connectons at once
        y0 = 1.0 / z0_ports
        y0_sum = jnp.dot(mask, y0) # Sums y0 for all ports sharing the same net
        y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
        X = mask * ((2.0 * y0_sqrt_prod) / y0_sum[:, None] - jnp.eye(topology.num_ports, dtype=s_block_diagonal.dtype))
        
        # Compute Intermediate matrix [T]
        # Because X aligns perfectly with the standard block-diagonal S matrix, 
        # we don't need to permute/reorder S into a C matrix.
        T = jnp.eye(topology.num_ports, dtype=s_block_diagonal.dtype) - S_trav @ X
        
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
        tmp_mat = jnp.linalg.pinv(T_D) @ T_C 
        
        numerator = X_A - X_B @ tmp_mat
        denominator = T_A - T_B @ tmp_mat
        S_ext_trav = numerator @ jnp.linalg.inv(denominator)
        
        # Convert back to power waves
        z0_ext = z0_ports[ext]
        S_ext_power = s2s(S_ext_trav, z0_ext, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_ext)