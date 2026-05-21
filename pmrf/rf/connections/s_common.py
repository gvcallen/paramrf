"""
Common S-parameter circuit connection algorithms.
"""

from typing import Sequence

import jax.numpy as jnp
from pmrf.utils.rf import fix_z0_shape


def connect_s_common(
    Smats: Sequence[jnp.ndarray] | jnp.ndarray,
    z0s: Sequence[jnp.ndarray] | jnp.ndarray,
    ports: Sequence[int | Sequence[int]],
) -> jnp.ndarray:
    """
    Connect a series of multi-port S-parameter matrices using Hallbjörner's method at a single intersection.
    
    Ensures that the specified port indices share the concatenated intersection, implying they are
    electrically common.

    Parameters
    ----------
    Smats : jnp.ndarray or Sequence[jnp.ndarray]
        S-parameter matrices. Shape of each matrix is `(Nf, n, n)`.
    z0s: jnp.ndarray or Sequence[jnp.ndarray]
        Characteristic impedances (z0) of each `S` in `Smats`.
    ports : Sequence[int | Sequence[int]]
        A sequence of port indices. Each entry corresponds to the ports of the respective 
        network in `Smats`. The length of `ports` should match the length of `Smats`.

    Returns
    -------
    tuple
        A tuple `(S, z0)` containing the combined S-matrix and z0 matrix.

    References
    ----------
    .. P. Hallbjörner, Microw. Opt. Technol. Lett. 38, 99 (2003).
    """
    # Adapted from scikit-rf. See the copyright notice in pmrf._frequency.py  

    # Handle single network input
    if isinstance(Smats, jnp.ndarray):
        Smats = [Smats]
    if isinstance(z0s, jnp.ndarray):
        z0s = [z0s]

    if len(Smats) != len(ports):
        raise ValueError(f'Smats and ports must have the same length ({len(Smats)} != {len(ports)})')
    if len(Smats) != len(z0s):
        raise ValueError(f'Smats and z0s must have the same length ({len(Smats)} != {len(z0s)})')

    # Get the index of each network in the list
    dim, off = sum(S.shape[1] for S in Smats), 0
    inter_indices, exter_indices =  [], []
    z0_in, z0_ext = [], []
    Nf = Smats[0].shape[0]

    # Assign the global scattering matrix [X] and concatenated intersection matrix [C]
    X = jnp.zeros((Nf, dim, dim), dtype='complex')
    C = jnp.zeros((Nf, dim, dim), dtype='complex')

    for i, (S, z0, port) in enumerate(zip(Smats, z0s, ports)):
        nports: int = S.shape[0]
        z0 = fix_z0_shape(z0)
        port = [port] if isinstance(port, int) else port

        # Check the port indecies valid or not
        if len(port) != len(set(port)):
            raise ValueError(f"Matrix {i}'s port should not be duplicated.")
        if max(port) >= nports or min(port) < 0:
            raise ValueError(f"Matrix {i}'s port index should be between 0 and {nports-1}")

        # Append the port index with offset to indices list
        for p in range(nports):
            if p in port:
                inter_indices.append(p + off)
                z0_in.append(z0[:, p])
            else:
                exter_indices.append(p + off)
                z0_ext.append(z0[:, p])

        # Assign the scattering matrix of each network to the global scattering matrix
        X[:, off:off+nports, off:off+nports] = S

        # Update the offset
        off += nports

    # Compute interaction matrix for internal connections
    z0s = jnp.array(z0_in).T
    y0s = 1./z0s
    y_tot = y0s.sum(axis=1)

    s = 2 *jnp.sqrt(jnp.einsum('ki,kj->kij', y0s, y0s)) / y_tot[:, None, None]
    jnp.einsum('kii->ki', s)[:] -= 1  # Sii

    # Get the index of internal port and external port from global matrix
    in_ind = jnp.meshgrid(inter_indices, inter_indices, indexing='ij')
    out_ind = jnp.meshgrid(exter_indices, exter_indices, indexing='ij')

    # Update the concatenated intersection matrix
    C[:, in_ind[0], in_ind[1]] = s

    # Get the global scattering matrix
    s = X @ jnp.linalg.inv(jnp.identity(dim) - C @ X)

    s_out = s[:, out_ind[0], out_ind[1]]

    z0_out = jnp.array(z0_ext).T
    return s_out, z0_out