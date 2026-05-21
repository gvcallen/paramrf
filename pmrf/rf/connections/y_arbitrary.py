"""
Arbitrary S-parameter circuit connection algorithms.
"""


import numpy as np
import jax.numpy as jnp
from typing import Sequence

def connect_y_arbitrary(
    Ymats: Sequence[jnp.ndarray],
    connections: Sequence[Sequence[tuple[int, int]]],
    port_indices: Sequence[int],
    grounded_nodes: Sequence[int] = None,
    method: str = 'nam',
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Connect multiple Y-parameter networks together in an arbitrary topology.

    Parameters
    ----------
    Ymats : Sequence[jnp.ndarray]
        A sequence of Y-parameter matrices for the component networks.
    connections : Sequence[Sequence[tuple[int, int]]]
        Topology connection nodes.
    port_indices : Sequence[int]
        External port network indices.
    grounded_nodes : Sequence[int] = None
        Grounded node indices.
    method : str, optional, default='nam'
        The algorithm to use.
    eps : float, optional, default=1e-12
        Tikhonov regularization parameter. Adds a microscopic conductance to 
        every node to prevent singular matrix inversions.

    Returns
    -------
    jnp.ndarray
        The resulting Y-parameter matrix.
    """
    if method == 'nam':
        return connect_y_arbitrary_nam(Ymats, connections, port_indices, grounded_nodes=grounded_nodes, eps=eps)
    else:
        raise ValueError(f"Unknown Y-parameter circuit connection method: {method}")


def connect_y_arbitrary_nam(
    Ymats: Sequence[jnp.ndarray],
    connections: Sequence[Sequence[tuple[int, int]]],
    port_indices: Sequence[int],
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Executes the Nodal Admittance Matrix (NAM) assembly and Schur complement.
    """
    if grounded_nodes is None:
        grounded_nodes = []

    Nf = Ymats[0].shape[0]
    num_nodes = len(connections)
    
    # 1. Map (network, port) to global node index
    port_to_node = {}
    for node_idx, cnx in enumerate(connections):
        for ntw_idx, port_idx in cnx:
            if ntw_idx not in port_to_node:
                port_to_node[ntw_idx] = {}
            port_to_node[ntw_idx][port_idx] = node_idx
            
    # 2. Precompute indices for scatter
    batch_Y_vals = []
    row_idx = []
    col_idx = []
    
    for ntw_idx, Y_mat in enumerate(Ymats):
        mapping = port_to_node[ntw_idx]
        num_ports = Y_mat.shape[1]
        for i in range(num_ports):
            for j in range(num_ports):
                batch_Y_vals.append(Y_mat[:, i, j])
                row_idx.append(mapping[i])
                col_idx.append(mapping[j])
                
    flat_Y = jnp.stack(batch_Y_vals, axis=-1)
    r_idx = np.array(row_idx)
    c_idx = np.array(col_idx)
    
    # 3. Assemble Global Matrix
    Y_global = jnp.zeros((Nf, num_nodes, num_nodes), dtype=Ymats[0].dtype)
    Y_global = Y_global.at[:, r_idx, c_idx].add(flat_Y)
    
    if eps > 0:
        Y_global = Y_global + (eps * jnp.eye(num_nodes, dtype=Y_global.dtype))
    
    # 4. Partition nodes into External, Internal, and Ground
    ext_nodes = []
    int_nodes = []
    
    for node_idx, cnx in enumerate(connections):
        if node_idx in grounded_nodes:
            continue # Skip grounded nodes entirely; they are deleted from the system
            
        if any(ntw_idx in port_indices for ntw_idx, _ in cnx):
            ext_nodes.append(node_idx)
        else:
            int_nodes.append(node_idx)
            
    ext = np.array(ext_nodes)
    int_n = np.array(int_nodes)
    
    # 5. Extract sub-matrices (implicitly dropping grounded rows/cols)
    Y_ee = Y_global[:, ext[:, None], ext]
    
    if len(int_nodes) > 0:
        Y_ei = Y_global[:, ext[:, None], int_n]
        Y_ie = Y_global[:, int_n[:, None], ext]
        Y_ii = Y_global[:, int_n[:, None], int_n]
        
        # Batched dense solve
        X = jnp.linalg.solve(Y_ii, Y_ie)
        Y_reduced = Y_ee - jnp.matmul(Y_ei, X)
    else:
        Y_reduced = Y_ee
        
    return Y_reduced