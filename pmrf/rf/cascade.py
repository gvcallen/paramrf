"""
Cascade reduction algorithms for series-connected 2N-port networks.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike


def _junction_inverse(M: jnp.ndarray, out: jnp.ndarray, rtol: float) -> jnp.ndarray:
    """
    Invert a junction matrix with its singular, unobservable directions pinned.

    A direction with ``M y = 0`` and ``out @ y = 0`` leaves the cascade unchanged
    whatever value it takes, so a rank-k update pins it and everything the
    external ports can observe is inverted exactly.
    """
    K = jax.lax.stop_gradient(jnp.concatenate((M, out), axis=0))
    _, sv, Vh = jnp.linalg.svd(K, full_matrices=False)
    R = Vh.conj().T * (sv <= rtol * sv[0])
    return jnp.linalg.inv(M + R @ R.conj().T)


def cascade_two_s(
    s_a: ArrayLike,
    z0_a: ArrayLike,
    s_b: ArrayLike,
    z0_b: ArrayLike,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Combine two 2N-port S-parameter matrices connected end-to-end.

    Ports $N$ to $2N-1$ of network $A$ are connected to ports $0$ to $N-1$ of
    network $B$. Both networks must share the same reference impedances across
    the connected interface; no renormalization is performed here.

    **Mathematical Formulation**

    Partitioning each network into $N \times N$ blocks, the combined network is

    $$S_{11} = A_{11} + A_{12} (I - B_{11} A_{22})^{-1} B_{11} A_{21}$$
    $$S_{12} = A_{12} (I - B_{11} A_{22})^{-1} B_{12}$$
    $$S_{21} = B_{21} (I - A_{22} B_{11})^{-1} A_{21}$$
    $$S_{22} = B_{22} + B_{21} (I - A_{22} B_{11})^{-1} A_{22} B_{12}$$

    Floating multi-conductor networks leave a common mode at the junction that
    makes $M = I - B_{11} A_{22}$ singular in theory, but only to rounding in
    practice. Inverting it, or pseudo-inverting at any fixed cutoff, amplifies
    that rounding into S and its gradients. Such a direction $y$ is also
    unobservable from the external ports, $A_{12} y = 0$, so the cascade does not
    depend on it. With $R$ an orthonormal basis of the directions where

    $$\begin{bmatrix} M \\ A_{12} \end{bmatrix} y \approx 0,$$

    judged by singular value relative to the largest, at a tolerance of
    $\sqrt{\epsilon}$ for the dtype, the combine inverts $M + R R^H$ exactly.
    The same applies to $N = I - A_{22} B_{11}$ with $B_{21}$. The basis is found
    outside autodiff, so gradients never pass through the zero singular values.
    A resonance is observable from the ports, so it is never pinned; for
    full-rank $M$ this is the plain inverse.

    A lossless network evaluated exactly at a true resonance frequency is still
    singular. That is physical, and the result there is not finite.

    Parameters
    ----------
    s_a : ArrayLike
        S-parameter matrix of the first network, shape (2N, 2N).
    z0_a : ArrayLike
        Port reference impedances of the first network, shape (2N,).
    s_b : ArrayLike
        S-parameter matrix of the second network, shape (2N, 2N).
    z0_b : ArrayLike
        Port reference impedances of the second network, shape (2N,).

    Returns
    -------
    s_cas : jnp.ndarray
        S-parameter matrix of the combined network, shape (2N, 2N).
    z0_cas : jnp.ndarray
        Port reference impedances of the combined network, shape (2N,).

    References
    ----------
    .. [1] Redheffer, R., "Difference Equations and Functional Equations in
           Transmission-Line Theory", in Modern Mathematics for the Engineer,
           2nd series, E. F. Beckenbach, Ed. McGraw-Hill, 1961, pp. 282-337.
           (The star product, of which this combine is the 2N-port form.)
    """
    Smat_A = jnp.asarray(s_a)
    Smat_B = jnp.asarray(s_b)
    z0_A = jnp.asarray(z0_a)
    z0_B = jnp.asarray(z0_b)

    nports = Smat_A.shape[0]
    N = nports // 2

    # Verify no un-renormalized impedance step exists between the stages
    mismatch_detected = jnp.any(jnp.abs(z0_A[N:] - z0_B[:N]) > 1e-6)
    Smat_A = eqx.error_if(
        Smat_A,
        mismatch_detected,
        "Scattering cascade requires matching reference impedances between connected ports. "
        "Renormalize stages or use a Circuit solver for arbitrary impedance steps."
    )

    z0_cas = jnp.concatenate((z0_A[:N], z0_B[N:]), axis=0)

    A11, A12 = Smat_A[:N, :N], Smat_A[:N, N:]
    A21, A22 = Smat_A[N:, :N], Smat_A[N:, N:]

    B11, B12 = Smat_B[:N, :N], Smat_B[:N, N:]
    B21, B22 = Smat_B[N:, :N], Smat_B[N:, N:]

    I = jnp.eye(N, dtype=Smat_A.dtype)

    rtol = np.sqrt(np.finfo(Smat_A.real.dtype).eps)
    X = _junction_inverse(I - B11 @ A22, A12, rtol)
    Y = _junction_inverse(I - A22 @ B11, B21, rtol)

    S11 = A11 + A12 @ X @ B11 @ A21
    S12 = A12 @ X @ B12
    S21 = B21 @ Y @ A21
    S22 = B22 + B21 @ Y @ A22 @ B12

    top = jnp.concatenate((S11, S12), axis=1)
    bottom = jnp.concatenate((S21, S22), axis=1)
    S_cas = jnp.concatenate((top, bottom), axis=0)

    return S_cas, z0_cas


def cascade_scattering(
    s_stacked: ArrayLike,
    z0_stacked: ArrayLike,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reduce a stack of 2N-port S-parameter matrices to a single cascaded network.

    Scans `cascade_two_s` across sequential sections, at a single frequency
    point. Use `jax.vmap` to apply it across a frequency axis.

    Parameters
    ----------
    s_stacked : ArrayLike
        S-parameter matrices of the sections, shape (M, 2N, 2N), ordered from
        the input of the cascade to its output.
    z0_stacked : ArrayLike
        Port reference impedances of the sections, shape (M, 2N).

    Returns
    -------
    s_cas : jnp.ndarray
        S-parameter matrix of the cascaded network, shape (2N, 2N).
    z0_cas : jnp.ndarray
        Port reference impedances of the cascaded network, shape (2N,).

    See Also
    --------
    cascade_two_s : The two-network combine this routine scans.
    """
    s_stacked = jnp.asarray(s_stacked)
    z0_stacked = jnp.asarray(z0_stacked)

    if s_stacked.shape[0] == 1:
        return s_stacked[0], z0_stacked[0]

    def scan_fn(carry, x):
        S_acc, z0_acc = carry
        S_i, z0_i = x
        S_next, z0_next = cascade_two_s(S_acc, z0_acc, S_i, z0_i)
        return (S_next, z0_next), None

    (S_cas, z0_cas), _ = jax.lax.scan(
        scan_fn,
        init=(s_stacked[0], z0_stacked[0]),
        xs=(s_stacked[1:], z0_stacked[1:])
    )
    return S_cas, z0_cas


def cascade_abcd(a_stacked: ArrayLike) -> jnp.ndarray:
    r"""
    Reduce a stack of ABCD-parameter matrices to a single cascaded network.

    The ABCD representation cascades by matrix multiplication, so the reduction
    is the ordered product of the sections, at a single frequency point. Use
    `jax.vmap` to apply it across a frequency axis.

    **Mathematical Formulation**

    $$A_{cas} = A_{0} A_{1} \cdots A_{M-1}$$

    Parameters
    ----------
    a_stacked : ArrayLike
        ABCD-parameter matrices of the sections, shape (M, 2N, 2N), ordered from
        the input of the cascade to its output.

    Returns
    -------
    jnp.ndarray
        ABCD-parameter matrix of the cascaded network, shape (2N, 2N).
    """
    a_stacked = jnp.asarray(a_stacked)

    if a_stacked.shape[0] == 1:
        return a_stacked[0]

    def scan_fn(carry, x):
        return carry @ x, None

    a_cas, _ = jax.lax.scan(
        scan_fn,
        init=a_stacked[0],
        xs=a_stacked[1:]
    )
    return a_cas
