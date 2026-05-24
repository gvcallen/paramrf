"""pmrf/simulate/topology.py"""

import equinox as eqx
from jaxtyping import ArrayLike
import numpy as np
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.simulate.base import PortRepresentation, NodalRepresentation


class Topology(eqx.Module):
    """A higher-level model containing sub-models and their topological connections."""
    
    models: list[Model]
    indexed_connections: list[list[tuple[int, int]]] = eqx.field(static=True)
    port_indices: list[int] = eqx.field(static=True)
    ground_indices: list[int] = eqx.field(static=True)
    
    def to_ports(self) -> PortRepresentation:
        """Generates the static S-Parameter topological representation."""
        num_ports = sum(m.nports for m in self.models)
        port_to_net_map = self.connected_components()
        
        # Assumes circuit.port_idxs contains flat global port indices
        ext_idx = np.array(self.port_indices or [], dtype=int)
        int_idx = np.setdiff1d(np.arange(num_ports), ext_idx)
        
        return PortRepresentation(
            num_ports=num_ports,
            ext_idx=ext_idx,
            int_idx=int_idx,
            port_to_net_map=port_to_net_map
        )    

    def to_nodal(self) -> NodalRepresentation:
        """Generates the static Y-Parameter topological representation using a dummy ground node."""
        port_to_net_map = self.connected_components()
        unique_nets = np.unique(port_to_net_map)
        
        # Identify which nets represent ground
        ground_nets = set()
        for p in (self.ground_indices or []):
            ground_nets.add(port_to_net_map[p])
            
        # Remap active nets to (0 ... V-1) and ground nets to a dummy node (V)
        num_active = 0
        remap = {}
        for net in unique_nets:
            if net not in ground_nets:
                remap[net] = num_active
                num_active += 1
                
        for net in ground_nets:
            remap[net] = num_active # Dummy ground node at the end
            
        final_port_nodes = np.array([remap[net] for net in port_to_net_map], dtype=int)
        
        # Build r_idx and c_idx by unrolling the local port matrices
        r_idx, c_idx = [], []
        offset = 0
        for m in self.models:
            n = m.nports
            nodes = final_port_nodes[offset:offset+n]
            for i in range(n):
                for j in range(n):
                    r_idx.append(nodes[i])
                    c_idx.append(nodes[j])
            offset += n
            
        # Identify external and internal active nets
        ext_nets = set()
        for p in (self.port_indices or []):
            net = final_port_nodes[p]
            if net != num_active: # Exclude the dummy ground node
                ext_nets.add(net)
                
        ext_idx = np.array(sorted(list(ext_nets)), dtype=int)
        all_active = set(range(num_active))
        int_idx = np.array(sorted(list(all_active - ext_nets)), dtype=int)
        
        return NodalRepresentation(
            num_nodes=num_active + 1, # +1 creates the space for the dummy ground node
            r_idx=np.array(r_idx, dtype=int),
            c_idx=np.array(c_idx, dtype=int),
            ext_idx=ext_idx,
            int_idx=int_idx
        )    
    
    def connected_components(self) -> np.ndarray:
        """
        Groups connected ports into unique nets using a Disjoint Set (Union-Find) algorithm.
        Returns a flat array mapping global_port_idx -> net_id.
        """
        def get_global_port(models: list[Model], m_idx: int, p_idx: int) -> int:
            return sum(m.nports for m in models[:m_idx]) + p_idx

        num_ports = sum(m.nports for m in self.models)
        parent = list(range(num_ports))
        
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        if self.indexed_connections:
            for cnx in self.indexed_connections:
                if not cnx: continue
                first = get_global_port(self.models, cnx[0][0], cnx[0][1])
                for m_idx, p_idx in cnx[1:]:
                    union(first, get_global_port(self.models, m_idx, p_idx))
                    
        port_to_net = np.array([find(i) for i in range(num_ports)], dtype=int)
        
        # Compress net IDs to contiguous integers (0 to num_unique_nets - 1)
        _, port_to_net_map = np.unique(port_to_net, return_inverse=True)
        return port_to_net_map
    
    def evaluate_scattering(self, freq: Frequency, z0: ArrayLike = 50.0, layout='block_diagonal') -> tuple[jnp.ndarray, jnp.ndarray]:
        if layout == 'block_diagonal':
            S_blocks = [m.s(freq, z0=z0) for m in self.models]
            
            Nf = S_blocks[0].shape[0]
            num_ports = sum(S.shape[1] for S in S_blocks)
            dtype = S_blocks[0].dtype
            
            # Assemble Block Diagonal Matrix
            batched_S = jnp.zeros((Nf, num_ports, num_ports), dtype=dtype)
            offset = 0
            for S_m in S_blocks:
                n = S_m.shape[1]
                batched_S = batched_S.at[:, offset:offset+n, offset:offset+n].set(S_m)
                offset += n
                
            # Safely handle z0 as scalar OR array by using broadcast_to instead of full
            batched_z0 = jnp.broadcast_to(jnp.asarray(z0, dtype=dtype), (num_ports,))
            return batched_S, batched_z0
            
        else:
            cascade_models = [
                m for m in self.models 
                if m.nports > 1 and type(m).__name__ not in ("Port", "Ground")
            ]
            
            # jnp.stack creates (N, F, m, m). 
            # We transpose to (F, N, m, m) so Frequency is ALWAYS axis 0.
            S_blocks = jnp.stack([m.s(freq, z0=z0) for m in cascade_models]).transpose(1, 0, 2, 3)
            
            n_models = len(cascade_models)
            m_ports = cascade_models[0].nports if n_models > 0 else 0
            batched_z0 = jnp.broadcast_to(jnp.asarray(z0), (n_models, m_ports))
            
            return S_blocks, batched_z0
                    

    def evaluate_admittance(self, freq: Frequency) -> jnp.ndarray:
        """Evaluates .y() on all models and flattens them for scatter-add assembly."""
        Y_blocks = [m.y(freq) for m in self.models]
        
        flat_Y_list = []
        for Y in Y_blocks:
            Nf, n, _ = Y.shape
            flat_Y_list.append(Y.reshape(Nf, n * n))
            
        batched_Y_elements = jnp.concatenate(flat_Y_list, axis=1)
        
        return batched_Y_elements    