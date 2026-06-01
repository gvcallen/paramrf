import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import List, Tuple

from pmrf.simulate.base import AbstractScatteringReducer, PortRepresentation, ScatteringResult

class BlockSchurScatteringReducer(AbstractScatteringReducer):
    """
    (experimental) Hierarchical Tree (Block Diakoptics) circuit solver for S-parameters.
    
    Eliminates internal nets in chunked multi-port clusters using block Schur 
    complements. This approach drastically minimizes cascading floating-point 
    errors by reducing the sequential JAX unroll depth, while keeping the 
    maximum matrix inversion size bounded.
    """
    #: The maximum number of connected port pairs to eliminate in a single Schur stage.
    #: A value of 4 means up to an 8x8 matrix inversion per step.
    max_pairs_per_stage: int = eqx.field(default=4, static=True)
    
    #: Numerical regularization to prevent singular matrix division during lossless resonance.
    eps: float = eqx.field(default=1e-12, static=True)

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        # Handle mathematically isolated external ports (VNA Probe Injection)
        S_curr, z0_curr, active_ext_idx, active_int_idx = self._inject_virtual_probes(
            s_block_diagonal, z0_ports, topology
        )
        
        # Extract pairs to connect and verify impedance matching
        pairs = self._extract_pairs(active_int_idx, topology.port_to_net_map)
        
        if pairs:
            k_idx = jnp.array([p[0] for p in pairs])
            l_idx = jnp.array([p[1] for p in pairs])
            z0_diff = jnp.abs(z0_curr[k_idx] - z0_curr[l_idx])
            mismatch_detected = jnp.any(z0_diff > 1e-6)
            
            S_curr = eqx.error_if(
                S_curr,
                mismatch_detected,
                "HierarchicalTreeReducer requires connected ports to have identical reference impedances."
            )

        # Partition pairs into hierarchical stages (Tree flattening)
        stages = [
            pairs[i : i + self.max_pairs_per_stage] 
            for i in range(0, len(pairs), self.max_pairs_per_stage)
        ]
        
        active_global_idx = list(range(S_curr.shape[0]))
        
        # Sequentially execute the multi-port Schur complement for each cluster
        for stage_pairs in stages:
            # Flatten pairs into a strict alternating list: [p1_a, p1_b, p2_a, p2_b, ...]
            elim_ports_global = [p for pair in stage_pairs for p in pair]
            
            # Map global indices to their current position in the shrinking active matrix
            elim_rel_idx = [active_global_idx.index(p) for p in elim_ports_global]
            keep_rel_idx = [i for i in range(len(active_global_idx)) if i not in elim_rel_idx]
            
            S_curr = self._multi_port_schur(
                S_curr, jnp.array(keep_rel_idx), jnp.array(elim_rel_idx)
            )
            
            # Update active index tracker
            active_global_idx = [active_global_idx[i] for i in keep_rel_idx]

        # Extract the final external S-parameters and impedances
        # The remaining indices perfectly align with the external indices
        final_ext_rel_idx = [active_global_idx.index(p) for p in active_ext_idx]
        
        S_ext = S_curr[jnp.ix_(final_ext_rel_idx, final_ext_rel_idx)]
        z0_ext = z0_curr[active_ext_idx]
        
        return ScatteringResult(s=S_ext, z0=z0_ext)

    def _multi_port_schur(self, S: jax.Array, keep_idx: jax.Array, elim_idx: jax.Array) -> jax.Array:
        """
        Eliminates a clustered subset of ports by solving the block S-parameter boundary condition.
        """
        S_ee = S[jnp.ix_(keep_idx, keep_idx)]
        S_ei = S[jnp.ix_(keep_idx, elim_idx)]
        S_ie = S[jnp.ix_(elim_idx, keep_idx)]
        S_ii = S[jnp.ix_(elim_idx, elim_idx)]
        
        K = elim_idx.shape[0]
        
        # Build Gamma connection matrix (Swaps adjacent nodes: 0<->1, 2<->3, 4<->5...)
        row = jnp.arange(K)
        col = row + jnp.where(row % 2 == 0, 1, -1)
        Gamma = jnp.zeros((K, K), dtype=S.dtype).at[row, col].set(1.0)
        
        # Construct Schur denominator: (I - S_ii * Gamma)
        M = jnp.eye(K, dtype=S.dtype) - S_ii @ Gamma
        
        if self.eps > 0:
            M += self.eps * jnp.eye(K, dtype=S.dtype)
            
        # X = M^-1 * S_ie
        X = jax.scipy.linalg.solve(M, S_ie, assume_a="gen")
        
        # S_new = S_ee + S_ei * Gamma * X
        S_new = S_ee + S_ei @ Gamma @ X
        
        return S_new

    @staticmethod
    def _inject_virtual_probes(
        s_matrix: jax.Array, z0: jax.Array, topology: PortRepresentation
    ) -> Tuple[jax.Array, jax.Array, np.ndarray, np.ndarray]:
        """
        Safely pads isolated external ports with virtual probes to prevent open-circuit failures.
        """
        ext_idx_np = np.array(topology.ext_idx)
        net_map_np = np.array(topology.port_to_net_map)
        
        net_counts = np.bincount(net_map_np)
        ext_net_counts = net_counts[net_map_np[ext_idx_np]]
        
        dangling_mask = ext_net_counts == 1
        num_dangling = np.sum(dangling_mask)
        
        if num_dangling == 0:
            return s_matrix, z0, ext_idx_np, np.array(topology.int_idx)
            
        N_orig = s_matrix.shape[-1]
        
        # Pad S-matrix and Z0
        pad_width = ((0, num_dangling), (0, num_dangling))
        s_padded = jnp.pad(s_matrix, pad_width, mode='constant')
        
        dangling_ext_indices = ext_idx_np[dangling_mask]
        z0_padded = jnp.concatenate([z0, z0[dangling_ext_indices]])
        
        # Update Topology Maps
        new_port_indices = np.arange(N_orig, N_orig + num_dangling)
        ext_idx_np[dangling_mask] = new_port_indices
        int_idx_np = np.concatenate([topology.int_idx, dangling_ext_indices])
        
        return s_padded, z0_padded, ext_idx_np, int_idx_np

    @staticmethod
    def _extract_pairs(int_idx: np.ndarray, port_to_net_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Pure Python pass to extract explicitly paired internal ports from the net map.
        """
        int_ports = int_idx.tolist()
        net_map = np.array(port_to_net_map)
        
        pairs = []
        visited = set()
        
        for p in int_ports:
            if p in visited:
                continue
                
            net_id = net_map[p]
            connected_ports = [x for x in int_ports if net_map[x] == net_id]
            
            if len(connected_ports) != 2:
                raise ValueError(
                    f"HierarchicalTreeReducer specifically requires pairs of ports. "
                    f"Net {net_id} has {len(connected_ports)} ports connected."
                )
                
            pairs.append(tuple(connected_ports))
            visited.update(connected_ports)
            
        return pairs