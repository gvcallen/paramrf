"""
Misc math functions.
"""
import jax.numpy as jnp
from jaxtyping import ArrayLike

NEAR_INF = 1e99

def rsolve(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    r"""
    Solves x @ A = B.

    Calls `jax.numpy.linalg.solve` with the last two axes swapped.

    Parameters
    ----------
    A : np.ndarray
        Matrix A.
    B : np.ndarray
        Matrix B.

    Returns
    -------
    x : np.ndarray
        Solution matrix.
    """
    A_H = jnp.matrix_transpose(A).conj()
    B_H = jnp.matrix_transpose(B).conj()
    return jnp.matrix_transpose(jnp.linalg.solve(A_H, B_H)).conj()

def nudge_diag(mat: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Stabilize matrix inversion by adding a tiny epsilon to the diagonal.
    Fully stable and smoothly differentiable in JAX.
    Supports unbatched (2D) and batched (N-D) square matrices.
    """
    # The last dimension is always the number of ports for a square matrix
    nports = mat.shape[-1]
    
    # Create an identity matrix for the inner 2D matrix
    Id = jnp.eye(nports, dtype=mat.dtype)
    
    # JAX will automatically broadcast this (nports, nports) identity matrix 
    # across any leading batch dimensions (like nfreqs) that `mat` might have.
    return mat + eps * Id


def unwrap_rad(phi: ArrayLike):
    """
    Unwrap a phase given in radians.

    Parameters
    ----------
    phi : number or array_like
        Phase in radians.

    Returns
    -------
    phi : number of array_like
        Unwrapped phase in radians.
    """
    return jnp.unwrap(phi, axis=0)