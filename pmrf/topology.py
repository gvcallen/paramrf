import equinox as eqx
from jaxtyping import ArrayLike
import numpy as np
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.base import Model


class Topology(eqx.Module):
    """
    A higher-level model containing sub-models and their topological connections.

    Attributes
    ----------
    models : list[Model]
        List of sub-models contained within this topology.
    indexed_connections : list[list[tuple[int, int]]]
        List of connection nets, where each net is a list of (model_index, port_index) tuples.
    port_indices : list[int]
        List of global port indices designated as external ports.
    ground_indices : list[int]
        List of global port indices designated as grounded connections.
    """
    
    models: list[Model]
    indexed_connections: list[list[tuple[int, int]]] = eqx.field(static=True)
    port_indices: list[int] = eqx.field(static=True)
    ground_indices: list[int] = eqx.field(static=True)
    
    def connected_components(self) -> np.ndarray:
        """
        Groups connected ports into unique nets using a Disjoint Set (Union-Find) algorithm.

        Returns
        -------
        np.ndarray
            A 1D array mapping each global port index to a contiguous, unique net ID.
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
    
    def evaluate_scattering(self, freq: Frequency, z0: ArrayLike = 50.0, layout: str = 'block_diagonal') -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluates the scattering parameters for all contained models.

        Parameters
        ----------
        freq : Frequency
            The frequency points at which to evaluate the models.
        z0 : ArrayLike, optional
            The reference impedance for the S-parameters, by default 50.0.
        layout : {'block_diagonal', 'stacked'}, optional
            The structural layout of the returned S-parameter tensor, by default 'block_diagonal'.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing:
            - batched_S: The S-parameters of the models structured according to the specified layout.
            - batched_z0: The corresponding reference impedances broadcasted to match the layout.

        Raises
        ------
        ValueError
            If an unsupported layout is provided.
        """
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
                
            batched_z0 = jnp.broadcast_to(jnp.asarray(z0, dtype=dtype), (num_ports,))
            return batched_S, batched_z0
            
        elif layout == 'stacked':
            cascade_models = []
            offset = 0
            marker_indices = set((self.port_indices or []) + (self.ground_indices or []))
            
            for m in self.models:
                # Exclude topological markers from the cascade solver list
                if m.nports > 1 and offset not in marker_indices:
                    cascade_models.append(m)
                offset += m.nports
            
            # jnp.stack creates (N, F, m, m). 
            # We transpose to (F, N, m, m) so Frequency is ALWAYS axis 0.
            S_blocks = jnp.stack([m.s(freq, z0=z0) for m in cascade_models]).transpose(1, 0, 2, 3)
            
            n_models = len(cascade_models)
            m_ports = cascade_models[0].nports if n_models > 0 else 0
            batched_z0 = jnp.broadcast_to(jnp.asarray(z0), (n_models, m_ports))
            
            return S_blocks, batched_z0
        else:
            raise ValueError(f"Unknown scattering layout: {layout}")
                    

    def evaluate_admittance(self, freq: Frequency, layout: str = 'flattened') -> jnp.ndarray:
        """
        Evaluates the admittance parameters for all models and structures them for assembly.

        Parameters
        ----------
        freq : Frequency
            The frequency points at which to evaluate the models.
        layout : {'flattened'}, optional
            The structural layout of the returned Y-parameter tensor, by default 'flattened'.

        Returns
        -------
        jnp.ndarray
            A 2D array of shape (N_frequencies, N_elements) containing the flattened 
            Y-parameter elements suitable for COO sparse matrix scatter-add operations.

        Raises
        ------
        ValueError
            If an unsupported layout is provided.
        """
        if layout != 'flattened':
            raise ValueError(f"Unknown admittance layout: {layout}")
        
        Y_blocks = []
        offset = 0
        
        # Pre-compute the set of global indices that belong to markers for O(1) lookup
        marker_indices = set((self.port_indices or []) + (self.ground_indices or []))
        
        for m in self.models:
            # If the model's global port offset is in the markers list, inject zeros
            if offset in marker_indices:
                zeros = jnp.zeros((freq.npoints, m.nports, m.nports), dtype=jnp.complex128)
                Y_blocks.append(zeros)
            else:
                Y_blocks.append(m.y(freq))
            
            offset += m.nports
            
        flat_Y_list = []
        for Y in Y_blocks:
            Nf, n, _ = Y.shape
            flat_Y_list.append(Y.reshape(Nf, n * n))
            
        batched_Y_elements = jnp.concatenate(flat_Y_list, axis=1)
        
        return batched_Y_elements

    def evaluate_impedance(self, freq: Frequency, layout: str = 'flattened') -> jnp.ndarray:
        """
        Evaluates the impedance parameters for all models and structures them for assembly.

        Parameters
        ----------
        freq : Frequency
            The frequency points at which to evaluate the models.
        layout : {'flattened'}, optional
            The structural layout of the returned Z-parameter tensor, by default 'flattened'.

        Returns
        -------
        jnp.ndarray
            A 2D array of shape (N_frequencies, N_elements) containing the flattened 
            Z-parameter elements suitable for COO sparse matrix scatter-add operations.

        Raises
        ------
        ValueError
            If an unsupported layout is provided.
        """
        if layout != 'flattened':
            raise ValueError(f"Unknown impedance layout: {layout}")
        
        Z_blocks = []
        offset = 0
        
        # Pre-compute the set of global indices that belong to markers for O(1) lookup
        marker_indices = set((self.port_indices or []) + (self.ground_indices or []))
        
        for m in self.models:
            # If the model's global port offset is in the markers list, inject zeros
            if offset in marker_indices:
                zeros = jnp.zeros((freq.npoints, m.nports, m.nports), dtype=jnp.complex128)
                Z_blocks.append(zeros)
            else:
                Z_blocks.append(m.z(freq))
            
            offset += m.nports
            
        flat_Z_list = []
        for Z in Z_blocks:
            Nf, n, _ = Z.shape
            flat_Z_list.append(Z.reshape(Nf, n * n))
            
        batched_Z_elements = jnp.concatenate(flat_Z_list, axis=1)
        
        return batched_Z_elements