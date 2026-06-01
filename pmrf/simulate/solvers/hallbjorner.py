"""pmrf/simulate/hallbjorner.py"""

import jax
import jax.numpy as jnp
import numpy as np

from pmrf.simulate.base import AbstractScatteringReducer, PortRepresentation, ScatteringResult
from pmrf.rf.conversions import s2s

class HallbjornerReducer(AbstractScatteringReducer):
    def run(
        self, 
        s_block_diagonal: jax.Array,  # Shape: (num_ports, num_ports) due to vmap
        z0_ports: jax.Array,          # Shape: (num_ports,)
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        # Virtual probe injection
        ext_idx_np = np.array(topology.ext_idx)
        net_map_np = np.array(topology.port_to_net_map)
        
        # Count how many ports are on each net
        net_counts = np.bincount(net_map_np)
        ext_net_counts = net_counts[net_map_np[ext_idx_np]]
        
        # An external port is "dangling" if it is the ONLY port on its net
        dangling_mask = ext_net_counts == 1
        num_dangling = np.sum(dangling_mask)
        
        if num_dangling > 0:
            N_orig = s_block_diagonal.shape[-1]
            
            # Pad the 2D S-matrix with 0.0s to represent perfect VNA sinks
            pad_width = ((0, num_dangling), (0, num_dangling))
            s_block_diagonal = jnp.pad(s_block_diagonal, pad_width, mode='constant')
            
            # Pad Z0 with the impedances of the corresponding dangling ports
            dangling_ext_indices = ext_idx_np[dangling_mask]
            z0_new = z0_ports[dangling_ext_indices]
            z0_ports = jnp.concatenate([z0_ports, z0_new])
            
            # Create indices for the newly injected virtual ports
            new_port_indices = np.arange(N_orig, N_orig + num_dangling)
            
            # Wire the virtual ports to the dangling nets
            net_map_np = np.concatenate([net_map_np, net_map_np[dangling_ext_indices]])
            
            # Move the previously dangling ports into the internal pool
            int_idx_np = np.concatenate([topology.int_idx, dangling_ext_indices])
            
            # Replace the dangling ports in the external pool with our virtual probes
            ext_idx_np[dangling_mask] = new_port_indices
        else:
            int_idx_np = np.array(topology.int_idx)
            
        # Reduction math
        S_trav = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        
        mask = jnp.array(
            net_map_np[:, None] == net_map_np[None, :], 
            dtype=s_block_diagonal.dtype
        )
        
        y0 = 1.0 / z0_ports
        y0_sum = jnp.dot(mask, y0) 
        y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
        X = mask * ((2.0 * y0_sqrt_prod) / y0_sum[:, None] - jnp.eye(net_map_np.shape[0], dtype=s_block_diagonal.dtype))
        
        T = jnp.eye(net_map_np.shape[0], dtype=s_block_diagonal.dtype) - S_trav @ X
        
        # Extract Sub-matrices using our augmented topology maps
        T_A = T[jnp.ix_(ext_idx_np, ext_idx_np)]
        T_B = T[jnp.ix_(ext_idx_np, int_idx_np)]
        T_C = T[jnp.ix_(int_idx_np, ext_idx_np)]
        T_D = T[jnp.ix_(int_idx_np, int_idx_np)]
        
        X_A = X[jnp.ix_(ext_idx_np, ext_idx_np)]
        X_B = X[jnp.ix_(ext_idx_np, int_idx_np)]
        
        # Schur Complement
        tmp_mat = jnp.linalg.pinv(T_D) @ T_C 
        
        numerator = X_A - X_B @ tmp_mat
        denominator = T_A - T_B @ tmp_mat
        S_ext_trav = numerator @ jnp.linalg.inv(denominator)
        
        z0_ext = z0_ports[ext_idx_np]
        S_ext_power = s2s(S_ext_trav, z0_ext, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_ext)