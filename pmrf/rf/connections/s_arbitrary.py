"""
Arbitrary S-parameter circuit connection algorithms.
"""

from typing import Sequence
from itertools import chain

import numpy as np
import jax.numpy as jnp
from pmrf.rf.conversions import s2s
from pmrf.utils.rf import fix_z0_shape

def connect_s_arbitrary(
    Smats: Sequence[jnp.ndarray],
    z0s: Sequence[jnp.ndarray],
    connections: Sequence[Sequence[tuple[int, int]]],
    port_indices: Sequence[int],
    s_def = 'power',
    method = 'hallbjorner2003',
) -> jnp.ndarray:
    """
    Connect multiple S-parameter networks together in an arbitrary topology.

    This is the most general function for connecting RF networks together in a circuit.

    This function constructs a global S-matrix by connecting ports from various
    sub-networks at common nodes (intersections).

    Parameters
    ----------
    Smats : Sequence[jnp.ndarray]
        A sequence of S-parameter matrices for the component networks.
        Each element has shape `(Nf, n_ports, n_ports)`.
    z0s : Sequence[jnp.ndarray]
        A sequence of characteristic impedances (z0) for the component networks.
    connections : Sequence[Sequence[tuple[int, int]]]
        A list of connection nodes. Each node is a list of `(network_index, port_index)`
        tuples indicating which ports are electrically connected at that node.
    port_indices : Sequence[int]
        A list of network indices. Any port belonging to a network in this list
        that appears in `connections` is treated as an external port of the
        resulting connected network.
    s_def : str, optional, default='power'
        The S-parameter definition to use ('power', 'traveling', or 'pseudo').
    method : str, optional, default='hallbjorner2003'
        The algorithm to use ('hallbjorner2003').

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        The resulting S-parameter matrix and characteristic impedances of the connected system.
    """
    if method == 'hallbjorner2003':
        return connect_s_arbitrary_hallbjorner(Smats, z0s, connections, port_indices=port_indices, s_def=s_def)
    else:
        raise Exception("Unknown S-parameter circuit connection method")
    

def connect_s_arbitrary_hallbjorner(
    Smats: Sequence[jnp.ndarray],
    z0s: Sequence[jnp.ndarray],
    connections: Sequence[Sequence[tuple[int, int]]],
    port_indices: Sequence[int],
    s_def = 'power',
) -> jnp.ndarray:
    """
    Connect multiple S-parameter networks together in an arbitrary topology.

    This is the most general function for connecting RF networks together in a circuit.

    This function constructs a global S-matrix by connecting ports from various
    sub-networks at common nodes (intersections).

    Parameters
    ----------
    Smats : Sequence[jnp.ndarray]
        A sequence of S-parameter matrices for the component networks.
        Each element has shape `(Nf, n_ports, n_ports)`.
    z0s : Sequence[jnp.ndarray]
        A sequence of characteristic impedances (z0) for the component networks.
    connections : Sequence[Sequence[tuple[int, int]]]
        A list of connection nodes. Each node is a list of `(network_index, port_index)`
        tuples indicating which ports are electrically connected at that node.
    port_indices : Sequence[int]
        A list of network indices. Any port belonging to a network in this list
        that appears in `connections` is treated as an external port of the
        resulting connected network.
    s_def : str, optional, default='power'
        The S-parameter definition to use ('power', 'traveling', or 'pseudo').

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        The resulting S-parameter matrix and characteristic impedances of the connected system.
    """
    dim = np.sum(np.array([len(cnx) for cnx in connections]))  # not JAX because dim is static
    Nf = Smats[0].shape[0]
    z0s = [fix_z0_shape(z0, Nf, S.shape[1]) for (S, z0) in zip(Smats, z0s)]

    port_connection_indices = []
    for (idx_cnx, (idx_ntw, _)) in enumerate(chain.from_iterable(connections)):
        if idx_ntw in port_indices:
            port_connection_indices.append(idx_cnx)

    in_idxs = [(i,) for i in range(dim) if i not in port_connection_indices]
    ext_idxs = [(i,) for i in port_connection_indices]
    ext_l, in_l = len(ext_idxs), len(in_idxs)

    # generate index slices for each sub-matrices
    idx_a, idx_b, idx_c, idx_d = (
        np.repeat(i, l, axis=1)
        for i, l in (
            (ext_idxs, ext_l),
            (ext_idxs, in_l),
            (in_idxs, ext_l),
            (in_idxs, in_l),
        )
    )

    # sub-matrices index, Matrix = [[A, B], [C, D]]]
    A_idx = (slice(None), idx_a, idx_a.T)
    B_idx = (slice(None), idx_b, idx_c.T)
    C_idx = (slice(None), idx_c, idx_b.T)
    D_idx = (slice(None), idx_d, idx_d.T)

    # Get the buffer of global matrix in f-order [X_T] and intermediate temporary matrix [T]
    # [T] = - [C] @ [X]
    networks = list(zip(Smats, z0s))
    
    x, t = connect_s_arbitrary_hallbjorner_X_and_T(networks, connections, port_connection_indices, s_def=s_def)

    # jnp.einsum('...ii->...i', t)[:] += 1
    t = t + jnp.eye(t.shape[-1])

    # Get the sub-matrices of inverse of intermediate temporary matrix t
    tmp_mat = jnp.linalg.pinv(t[D_idx]) @ t[C_idx]

    # Get the external S-parameters for the external ports
    # Calculated by multiplying the sub-matrices of x and t
    S_ext = (x[A_idx] - x[B_idx] @ tmp_mat) @ jnp.linalg.inv(
        t[A_idx] - t[B_idx] @ tmp_mat
    )

    z0s = []
    for cnx in connections:
        for (ntw_idx, ntw_port) in cnx:
            z0s.append(networks[ntw_idx][1][:,ntw_port])
    port_z0 = jnp.array(z0s)[port_connection_indices, :].T  # shape (nb_freq, nb_ports)    

    S_ext_converted = s2s(S_ext, port_z0, s_def, 'traveling')
    return S_ext_converted, port_z0

def connect_s_arbitrary_hallbjorner_C(
    networks: Sequence[tuple[jnp.ndarray, jnp.ndarray]],
    connections: Sequence[Sequence[tuple[int, int]]],
    port_connection_indices: Sequence[int],
    s_def = 'power',    
):
    """
    Construct the connection scattering matrix [C] for connect_s_arbitrary.

    Parameters
    ----------
    networks : Sequence[tuple[jnp.ndarray, jnp.ndarray]]
        A sequence of (S-parameter, z0) tuples for the component networks.
    connections : Sequence[Sequence[tuple[int, int]]]
        A list of connections, where each connection is a list of (network_index, port_index) tuples
        representing ports connected at a single node.
    port_connection_indices : Sequence[int]
        Indices of the ports in the flattened connection list that are designated as external ports.
    s_def : str, default='power'
        The S-parameter definition to use ('power', 'traveling', or 'pseudo').

    Returns
    -------
    jnp.ndarray
        The connection scattering matrix with shape `(Nf, dim, dim)`.
    """
    Nf = networks[0][0].shape[0]
    dim = np.sum(np.array([len(cnx) for cnx in connections]))  # not JAX because dim is static
    connections_list = list(enumerate(chain.from_iterable(connections)))

    # list all network indices which are not considered "ports"
    internal_connection_indices = [k for k in range(len(networks)) if k not in port_connection_indices]

    # generate the port reordering indexes from each connections
    ntws_ports_reordering = {ntwk_idx:[] for ntwk_idx in internal_connection_indices}
    for (cnx_idx, (ntwk_idx, port)) in connections_list:
        if ntwk_idx in ntws_ports_reordering:
            ntws_ports_reordering[ntwk_idx].append([port, cnx_idx])

    # re-ordering scattering parameters
    S = jnp.zeros((Nf, dim, dim), dtype='complex')

    for (ntwk_idx, ports) in ntws_ports_reordering.items():
        # get the port re-ordering indexes (from -> to)
        ports = np.array(ports)

        # port permutations
        from_port = ports[:,0]
        to_port = ports[:,1]

        for (_from, _to) in zip(from_port, to_port):
            S_network, z0_network = networks[ntwk_idx]
            S_converted = s2s(S_network, z0_network, 'traveling', s_def)
            S = S.at[:, _to, to_port].set(S_converted[:, _from, from_port])

    return S  # shape (nb_frequency, nb_inter*nb_n, nb_inter*nb_n)

def connect_s_arbitrary_hallbjorner_X(
    networks: Sequence[tuple[jnp.ndarray, jnp.ndarray]],
    connections: Sequence[Sequence[tuple[int, int]]],        
    inverse: bool = False,
    s_def = 'power',    
) -> jnp.ndarray:
    """
    Construct the block diagonal matrix [X] of component S-matrices, adjusted for node impedances.

    Parameters
    ----------
    networks : Sequence[tuple[jnp.ndarray, jnp.ndarray]]
        A sequence of (S-parameter, z0) tuples.
    connections : Sequence[Sequence[tuple[int, int]]]
        Topology definitions.
    inverse : bool, optional, default=False
        If True, return the conjugate of X.
    s_def : str, default='power'
        The S-parameter definition.

    Returns
    -------
    jnp.ndarray
        The global X matrix with shape `(Nf, dim, dim)`.
    """
    def _Xk(cnx_k):
        y0s = jnp.array([1/networks[idx][1][:,port] for (idx, port) in cnx_k]).T
        y_k = y0s.sum(axis=1)
        Nf = networks[0][0].shape[0]

        Xs = jnp.zeros((Nf, len(cnx_k), len(cnx_k)), dtype='complex')
        Xs = 2 *jnp.sqrt(jnp.einsum('ij,ik->ijk', y0s, y0s)) / y_k[:, None, None]
        # jnp.einsum('kii->ki', Xs)[:] -= 1  # Sii
        Xs = Xs.at[:, jnp.arange(Xs.shape[1]), jnp.arange(Xs.shape[1])].add(-1)

        return jnp.conjugate(Xs) if inverse else Xs        

    Xks = [_Xk(cnx) for cnx in connections]
    Nf = networks[0][0].shape[0]
    dim = np.sum(np.array([len(cnx) for cnx in connections]))  # not JAX because dim is static

    Xf = jnp.zeros((Nf, dim, dim), dtype='complex')
    off = np.array([0, 0])
    for Xk in Xks:
        # Xf[:, off[0]:off[0] + Xk.shape[1], off[1]:off[1]+Xk.shape[2]] = Xk
        Xf = Xf.at[:, off[0]:off[0] + Xk.shape[1], off[1]:off[1] + Xk.shape[2]].set(Xk)
        off += Xk.shape[1:]

    return Xf 
    
def connect_s_arbitrary_hallbjorner_X_and_T(
    networks: Sequence[tuple[jnp.ndarray, jnp.ndarray]],
    connections: Sequence[Sequence[tuple[int, int]]], 
    port_connection_indices: Sequence[int],       
    s_def = 'power',    
):
    """
    Compute the [X] and [T] matrices required for the connection algorithm.

    Parameters
    ----------
    networks : Sequence[tuple[jnp.ndarray, jnp.ndarray]]
        Component networks and impedances.
    connections : Sequence[Sequence[tuple[int, int]]]
        Topology connections.
    port_connection_indices : Sequence[int]
        Indices of external ports.
    s_def : str, default='power'
        S-parameter definition.

    Returns
    -------
    tuple
        A tuple `(X, T)` of JAX arrays.
    """
    X = connect_s_arbitrary_hallbjorner_X(networks, connections, s_def=s_def)
    C = connect_s_arbitrary_hallbjorner_C(networks, connections, port_connection_indices, s_def=s_def)

    # T will be fully updated, so `empty_like` is safe and faster
    T = jnp.empty_like(X, dtype="complex")

    # Precompute the sizes of connections and slices for each intersection
    cnx_size = [len(cnx) for cnx in connections]
    slice_ = np.cumsum([0] + cnx_size)
    slices = [slice(slice_[i], slice_[i+1]) for i in range(len(cnx_size))]

    # Perform the multiplication
    for j_slice in slices:
        # Get the Block diagonal part of X and corresponding C matrix buffer
        X_jj = - X[:, j_slice, j_slice]
        C_j = C[:, :, j_slice]

        # Perform the multiplication
        for i_slice in slices:
            # T[:, i_slice, j_slice] = C_j[:, i_slice, :] @ X_jj
            T = T.at[:, i_slice, j_slice].set(C_j[:, i_slice, :] @ X_jj)

    return X, T