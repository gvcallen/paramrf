"""
Cascade algorithms.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike

from pmrf.utils.rf import fix_z0_shape
from pmrf.math import nudge_diag

@eqx.filter_jit
def cascade_s(
    Smats: ArrayLike,
    z0s: ArrayLike,
    method = 'redheffer',
    eps=1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cascades multiple S-parameter networks together.

    This is the most efficient function to use for cascading S-parameter networks together.

    Currently only the 2N port case is supported, in which case Redheffer's star product is used.
    Also, currently, all matrices must have the same S-parameter definition and characteristic impedance.

    Parameters
    ----------
    Smats : ArrayLike
        A sequence of S-parameter matrices for the component networks.
        The array must have shape (N x F x m x m) representing N networks to be
        cascaded sequential at F frequencies, each with ports (m, m).
    z0s : ArrayLike
        A sequence of characteristic impedances (z0) for the component networks.
        Must be a float, or N arrays each broadcastable to (N x F x m).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        The resulting S-parameter matrix and characteristic impedance of the cascaded system.
    """
    if method != 'redheffer':
        raise ValueError("Currently only method = 'redheffer' is supported for cascade_s")

    Smats = jnp.array(Smats)
    z0s = jnp.array(z0s)

    # Fast path for single network
    if Smats.shape[0] == 1:
        return Smats[0], z0s[0]

    # Check port shapes and impedances
    nnetworks = Smats.shape[0]
    if not all(s.shape == Smats[0].shape for s in Smats):
        raise ValueError("Currently, all networks must have the same number of ports")
    nfreq, nports, nports = Smats[0].shape

    z0s = fix_z0_shape(z0s, nfreq, nports, nnetworks)
    z0s = eqx.error_if(
        z0s, 
        jnp.logical_not(jnp.all(z0s == z0s[0])), 
        "Currently, all characteristic impedances must be equal in cascade_s"
    )    

    # Connect all S-matrices using jax.lax.scan
    def scan_fn(carry, x):
        S_acc, z0_acc = carry
        S_i, z0_i = x
        S_next, z0_next = _cascade_two_s_redheffer(S_acc, z0_acc, S_i, z0_i, eps=eps)
        return (S_next, z0_next), None

    # jax.lax.scan loops over the arrays dynamically without unrolling
    (S_cas, z0_cas), _ = jax.lax.scan(
        scan_fn, 
        init=(Smats[0], z0s[0]), 
        xs=(Smats[1:], z0s[1:])
    )
    
    return S_cas, z0_cas
    
def _cascade_two_s_redheffer(
    Smat_A: jnp.ndarray,
    z0_A: jnp.ndarray,
    Smat_B: jnp.ndarray,
    z0_B: jnp.ndarray,
    eps=1e-12,
):
    nports = Smat_A.shape[1]
    assert nports % 2 == 0
    N = nports // 2

    z0_cas = jnp.concatenate((z0_A[:, :N], z0_B[:, N:]), axis=1)

    # Partition blocks
    A11 = Smat_A[:, :N, :N]
    A12 = Smat_A[:, :N, N:]
    A21 = Smat_A[:, N:, :N]
    A22 = Smat_A[:, N:, N:]

    B11 = Smat_B[:, :N, :N]
    B12 = Smat_B[:, :N, N:]
    B21 = Smat_B[:, N:, :N]
    B22 = Smat_B[:, N:, N:]

    I = jnp.broadcast_to(jnp.eye(N), (Smat_A.shape[0], N, N))

    M = nudge_diag(I - B11 @ A22, eps=eps)
    N = nudge_diag(I - A22 @ B11, eps=eps)
    X = jnp.linalg.solve(M, I)
    Y = jnp.linalg.solve(N, I)

    S11 = A11 + A12 @ X @ B11 @ A21
    S12 = A12 @ X @ B12
    S21 = B21 @ Y @ A21
    S22 = B22 + B21 @ Y @ A22 @ B12

    top = jnp.concatenate((S11, S12), axis=2)
    bottom = jnp.concatenate((S21, S22), axis=2)

    S_cas = jnp.concatenate((top, bottom), axis=1)

    return S_cas, z0_cas

def cascade_a(
    Amats: Sequence[jnp.ndarray],
) -> jnp.ndarray:
    """
    Cascaded multiple ABCD-parameter networks together in a loop.

    This simply multiplies the matrices together.

    Parameters
    ----------
    Smats : Sequence[jnp.ndarray]
        A sequence of S-parameter matrices for the component networks.
        Each element has shape `(Nf, n_ports, n_ports)`.
    z0s : Sequence[jnp.ndarray]
        A sequence of characteristic impedances (z0) for the component networks.
    s_def : str, optional, default='power'
        The S-parameter definition to use ('power', 'traveling', or 'pseudo').

    Returns
    -------
    jnp.ndarray
        The resulting S-parameter matrix of the cascaded system.
    """
    # Ensure we are working with a stacked array of shape (N, Nf, 2, 2)
    Amats_array = jnp.asarray(Amats)
    
    # Fast path for a single network
    if Amats_array.shape[0] == 1:
        return Amats_array[0]

    def scan_fn(carry, x):
        # The '@' operator natively broadcasts across the frequency dimension (Nf)
        return carry @ x, None

    # Scan sequentially multiplies the matrices across the 'N' dimension
    final_a, _ = jax.lax.scan(scan_fn, init=Amats_array[0], xs=Amats_array[1:])
    
    return final_a

def cascade_t(
    Tmats: Sequence[jnp.ndarray],
) -> jnp.ndarray:
    """
    Cascade multiple wave-transfer (T) matrices.

    Parameters
    ----------
    Tmats : Sequence[jnp.ndarray]
        Sequence of T matrices with shape (Nf, 2, 2)

    Returns
    -------
    jnp.ndarray
        Cascaded T matrix with shape (Nf, 2, 2)
    """

    Tmats_array = jnp.asarray(Tmats)

    if Tmats_array.shape[0] == 1:
        return Tmats_array[0]

    def scan_fn(carry, x):
        return carry @ x, None

    final_t, _ = jax.lax.scan(scan_fn, init=Tmats_array[0], xs=Tmats_array[1:])

    return final_t