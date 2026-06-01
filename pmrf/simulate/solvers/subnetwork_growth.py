import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from pmrf.simulate.base import AbstractScatteringReducer, PortRepresentation, ScatteringResult

class SubnetworkGrowthReducer(AbstractScatteringReducer):
    """
    Iterative Sub-Network Growth (Port Elimination) algorithm in JAX.
    Connects internal ports in pairs, eliminating them algebraically.

    More efficient than global algorithms for a small number of connections,
    but scales poorly for larger networks due to sequential graph unrolling.
    """
    eps: float = eqx.field(default=1e-12, static=True)
    
    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        # Extract pairs to connect safely at trace-time
        pairs_to_connect = self._extract_pairs(topology)
        
        if pairs_to_connect:
            # Statically extract the global indices for all connected pairs
            k_idx = jnp.array([p[0] for p in pairs_to_connect])
            l_idx = jnp.array([p[1] for p in pairs_to_connect])
            
            # Check for impedance mismatches (allowing 1e-6 for floating-point drift)
            z0_diff = jnp.abs(z0_ports[k_idx] - z0_ports[l_idx])
            mismatch_detected = jnp.any(z0_diff > 1e-6)
            
            # eqx.error_if binds the check to a data dependency to ensure JAX evaluates it
            s_block_diagonal = eqx.error_if(
                s_block_diagonal,
                mismatch_detected,
                "SubnetworkGrowthReducer requires connected ports to have identical reference impedances."
            )
        # ---------------------------------------------------------
        
        S_current = s_block_diagonal
        active_indices = list(range(topology.num_ports))
        
        # Sequentially unrolled during JAX JIT compilation
        for k_global, l_global in pairs_to_connect:
            k_curr = active_indices.index(k_global)
            l_curr = active_indices.index(l_global)
            
            S_current = self._pairwise_connect(S_current, k_curr, l_curr)
            
            active_indices.pop(max(k_curr, l_curr))
            active_indices.pop(min(k_curr, l_curr))

        z0_ext = z0_ports[topology.ext_idx]
        
        return ScatteringResult(s=S_current, z0=z0_ext)

    @staticmethod
    def _extract_pairs(topology: PortRepresentation) -> list[tuple[int, int]]:
        """
        Pure Python logic to pair up internal ports based on their net IDs.
        """
        int_ports = topology.int_idx.tolist()
        net_map = np.array(topology.port_to_net_map)
        
        pairs = []
        visited = set()
        
        for p in int_ports:
            if p in visited:
                continue
                
            net_id = net_map[p]
            connected_ports = [x for x in int_ports if net_map[x] == net_id]
            
            if len(connected_ports) != 2:
                raise ValueError(
                    f"Sub-network growth specifically requires pairs of ports. "
                    f"Net {net_id} has {len(connected_ports)} ports connected."
                )
                
            pairs.append(tuple(connected_ports))
            visited.update(connected_ports)
            
        return pairs

    def _pairwise_connect(self, S: jax.Array, k: int, l: int) -> jax.Array:
        """
        Core algebraic reduction. Eliminates ports k and l from the S-matrix.
        Returns a newly condensed matrix of shape (N-2, N-2).
        """
        N = S.shape[0]
        ext_idx = jnp.array([i for i in range(N) if i != k and i != l])
        
        Akl = S[k, l]
        Alk = S[l, k]
        Akk = S[k, k]
        All = S[l, l]
        
        det = (1.0 - Akl) * (1.0 - Alk) - Akk * All
        
        # Prevent division by zero during lossless resonance
        det = jnp.where(jnp.abs(det) < self.eps, self.eps, det)
        
        Ake = S[k, ext_idx]
        Ale = S[l, ext_idx]
        Aek = S[ext_idx, k]
        Ael = S[ext_idx, l]
        
        tmp_a = Ael * ((1.0 - Alk) / det) + Aek * (All / det)
        tmp_b = Ael * (Akk / det) + Aek * ((1.0 - Akl) / det)
        
        # update_matrix = jnp.outer(Ake, tmp_a) + jnp.outer(Ale, tmp_b)
        update_matrix = jnp.outer(tmp_a, Ake) + jnp.outer(tmp_b, Ale)
        S_reduced = S[jnp.ix_(ext_idx, ext_idx)] + update_matrix
        
        return S_reduced