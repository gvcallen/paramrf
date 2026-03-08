from typing import Any

import jax.numpy as jnp
import jax
from jax import lax
from jax.scipy.special import gammaln
from jax._src.numpy.ufuncs import _constant_like

from pmrf.constants import NumberLike
from pmrf.math_functions import rsolve, nudge_eig, nudge_diag
from pmrf.rf_functions.normalize import fix_z0_shape

ZERO = 1e-4

def s2s(s: NumberLike, z0: NumberLike, s_def_new, s_def_old):
    """
    Convert S-parameters between different definitions (e.g., Power waves vs Traveling waves).

    This function handles the conversion logic defined by `s_def_old` to `s_def_new`.
    It supports complex characteristic impedances.

    Parameters
    ----------
    s : jnp.ndarray
        The S-parameter matrix with shape `(nfreqs, nports, nports)`.
    z0 : NumberLike
        The characteristic impedance. Can be a scalar, or an array broadcastable
        to `(nfreqs, nports)`.
    s_def_new : str
        The target S-parameter definition. Options: 'power', 'traveling'.
    s_def_old : str
        The source S-parameter definition. Options: 'power', 'traveling'.

    Returns
    -------
    jnp.ndarray
        The converted S-parameter matrix with shape `(nfreqs, nports, nports)`.
    """
    if s_def_new == s_def_old:
        return s

    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)

    all_real = jnp.isreal(z0).all()
    
    def real_branch():
        return s
    
    def imag_branch():
        # Calculate port voltages and currents using the old s_def.
        F, G = jnp.zeros_like(s), jnp.zeros_like(s)
        diag_idx = jnp.arange(F.shape[1])
        if s_def_old == 'power':
            F = F.at[:, diag_idx, diag_idx].set(1.0 / (jnp.sqrt(z0.real)))
            G = G.at[:, diag_idx, diag_idx].set(z0)        
            Id = jnp.eye(s.shape[1], dtype=complex)
            v = F @ (G.conjugate() + G @ s)
            i = F @ (Id - s)
        elif s_def_old == 'traveling':
            F = F.at[:, diag_idx, diag_idx].set(jnp.sqrt(z0))
            G = G.at[:, diag_idx, diag_idx].set(1/(jnp.sqrt(z0)))        
            Id = jnp.eye(s.shape[1], dtype=complex)
            v = F @ (Id + s)
            i = G @ (Id - s)
        else:
            raise ValueError(f'Unknown s_def: {s_def_old}')

        # Calculate a and b waves from the voltages and currents.
        F, G = jnp.zeros_like(s), jnp.zeros_like(s)
        if s_def_new == 'power':
            F = F.at[:, diag_idx, diag_idx].set(1/(2*jnp.sqrt(z0.real)))
            G = G.at[:, diag_idx, diag_idx].set(z0)     
            a = F @ (v + G @ i)
            b = F @ (v - G.conjugate() @ i)
        elif s_def_new == 'traveling':
            F = F.at[:, diag_idx, diag_idx].set(1/(jnp.sqrt(z0)))
            G = G.at[:, diag_idx, diag_idx].set(z0) 
            a = F @ (v + G @ i)
            b = F @ (v - G @ i)
        else:
            raise ValueError(f'Unknown s_def: {s_def_old}')

        # New S-parameter matrix from a and b waves.
        s_new = jnp.zeros_like(s)
        for n in range(nports):
            for m in range(nports):
                s_new = s_new.at[:, m, n].set(b[:, m, n] / a[:, n, n])

        return s_new

    return jax.lax.cond(all_real, real_branch, imag_branch)

def a2s(a: jnp.ndarray, z0: NumberLike = 50) -> jnp.ndarray:
    """
    Convert ABCD parameters to S-parameters.

    Parameters
    ----------
    a : jnp.ndarray
        The ABCD parameter matrix with shape `(nfreqs, 2, 2)`.
    z0 : NumberLike, optional, default=50
        The characteristic impedance.

    Returns
    -------
    jnp.ndarray
        The S-parameter matrix with shape `(nfreqs, 2, 2)`.

    Raises
    ------
    IndexError
        If the input is not a 2-port network.
    """
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    nfreqs, nports, nports = a.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    A = a[:,0,0]
    B = a[:,0,1]
    C = a[:,1,0]
    D = a[:,1,1]
    denom = A*z02 + B + C*z01*z02 + D*z01

    s = jnp.array([
        [
            (A*z02 + B - C*z01.conj()*z02 - D*z01.conj() ) / denom,
            (2*jnp.sqrt(z01.real * z02.real)) / denom,
        ],
        [
            (2*(A*D - B*C)*jnp.sqrt(z01.real * z02.real)) / denom,
            (-A*z02.conj() + B - C*z01*z02.conj() + D*z01) / denom,
        ],
    ]).transpose()
    return s

def s2a(s: jnp.ndarray, z0: NumberLike = 50) -> jnp.ndarray:
    """
    Convert S-parameters to ABCD parameters.

    Parameters
    ----------
    s : jnp.ndarray
        The S-parameter matrix with shape `(nfreqs, 2, 2)`.
    z0 : NumberLike, optional, default=50
        The characteristic impedance.

    Returns
    -------
    jnp.ndarray
        The ABCD parameter matrix with shape `(nfreqs, 2, 2)`.

    Raises
    ------
    IndexError
        If the input is not a 2-port network.
    """
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    nfreqs, nports, nports = s.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    denom = (2*s[:,1,0]*jnp.sqrt(z01.real * z02.real))
    a = jnp.array([
        [
            ((z01.conj() + s[:,0,0]*z01)*(1 - s[:,1,1]) + s[:,0,1]*s[:,1,0]*z01) / denom,
            ((1 - s[:,0,0])*(1 - s[:,1,1]) - s[:,0,1]*s[:,1,0]) / denom,
        ],
        [
            ((z01.conj() + s[:,0,0]*z01)*(z02.conj() + s[:,1,1]*z02) - s[:,0,1]*s[:,1,0]*z01*z02) / denom,
            ((1 - s[:,0,0])*(z02.conj() + s[:,1,1]*z02) + s[:,0,1]*s[:,1,0]*z02) / denom,
        ],
    ]).transpose()
    return a

def s2y(s: jnp.ndarray, z0: NumberLike = 50, s_def: str = 'power') -> jnp.ndarray:
    """
    Convert S-parameters to Admittance (Y) parameters.

    Parameters
    ----------
    s : jnp.ndarray
        The S-parameter matrix with shape `(nfreqs, nports, nports)`.
    z0 : NumberLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The Admittance matrix with shape `(nfreqs, nports, nports)`.
    """
    nfreqs, nports, _ = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    z0 = z0.astype(dtype=complex)
    z0 = jnp.where(z0.real == 0, z0 + ZERO, z0)

    s = jnp.array(s, dtype=complex)

    # Creating Identity matrices of shape (nports,nports) for each nfreqs
    Id = jnp.eye(nports, dtype=complex)[None, :, :]
    Id = jnp.broadcast_to(Id, (nfreqs, nports, nports))

    if s_def == 'power':
        F, F_inv, G = jnp.zeros_like(s), jnp.zeros_like(s), jnp.zeros_like(s)
        diag_idx = jnp.arange(nports)
        
        # F_inv is the inverse of F: a diagonal matrix of 2 * sqrt(Re(Z0))
        F = F.at[:, diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0.real)))
        F_inv = F_inv.at[:, diag_idx, diag_idx].set(2 * jnp.sqrt(z0.real))
        G = G.at[:, diag_idx, diag_idx].set(z0)
        
        # Left-solve: X = A^-1 B  =>  jnp.linalg.solve(A, B)
        # Y = F_inv @ (S @ G + G^*)^-1 @ (I - S) @ F
        A = s @ G + jnp.conjugate(G)
        B = Id - s
        
        y = F_inv @ jnp.linalg.solve(nudge_diag(A), B) @ F

    elif s_def == 'traveling':
        # Creating diagonal matrices of 1 / sqrt(Z0)
        inv_sqrtz0 = jnp.zeros_like(s)
        jnp.einsum('ijj->ij', inv_sqrtz0)[...] = 1.0 / jnp.sqrt(z0)
        
        # Y = Z0^-1/2 @ (I + S)^-1 @ (I - S) @ Z0^-1/2
        A = Id + s
        B = Id - s
        
        y = inv_sqrtz0 @ jnp.linalg.solve(nudge_diag(A), B) @ inv_sqrtz0

    else:
        raise ValueError(f"Unknown s_def: {s_def}")

    return y

def y2s(y: jnp.ndarray, z0: NumberLike = 50, s_def = 'power') -> jnp.ndarray:
    """
    Convert Admittance (Y) parameters to S-parameters.

    Parameters
    ----------
    y : jnp.ndarray
        The Admittance matrix with shape `(nfreqs, nports, nports)`.
    z0 : NumberLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The S-parameter matrix with shape `(nfreqs, nports, nports)`.
    """
    nfreqs, nports, nports = y.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    z0 = z0.astype(dtype=complex)
    z0 = jnp.where(z0.real == 0, z0 + ZERO, z0)

    y = jnp.array(y, dtype=complex)

    # The following is a vectorized version of a for loop for all frequencies.
    # Creating Identity matrices of shape (nports,nports) for each nfreqs
    Id = jnp.eye(nports, dtype=complex)[None, :, :]  # (1, nports, nports)
    Id = jnp.broadcast_to(Id, (nfreqs, nports, nports))

    if s_def == 'power':
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F, G = jnp.zeros_like(y), jnp.zeros_like(y)
        diag_idx = jnp.arange(F.shape[1])
        F = F.at[:, diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0.real)))
        G = G.at[:, diag_idx, diag_idx].set(z0)        
        s = rsolve(F @ (Id + G @ y), F @ (Id - jnp.conjugate(G) @ y))
    elif s_def == 'traveling':
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating diagonal matrices of shape (nports, nports) for each nfreqs
        sqrtz0 = jnp.zeros_like(y)  # (nfreqs, nports, nports)
        jnp.einsum('ijj->ij', sqrtz0)[...] = jnp.sqrt(z0)
        s = rsolve(Id + sqrtz0 @ y @ sqrtz0, Id - sqrtz0 @ y @ sqrtz0)
    else:
        raise ValueError(f'Unknown s_def: {s_def}')

    return s

def s2z(s: jnp.ndarray, z0: NumberLike = 50, s_def = 'power') -> jnp.ndarray:
    """
    Convert S-parameters to Impedance (Z) parameters.

    Parameters
    ----------
    s : jnp.ndarray
        The S-parameter matrix with shape `(nfreqs, nports, nports)`.
    z0 : NumberLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The Impedance matrix with shape `(nfreqs, nports, nports)`.
    """
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    z0 = z0.astype(dtype=complex)
    z0 = jnp.where(z0.real == 0, z0 + ZERO, z0)

    s = jnp.array(s, dtype=complex)

    # The following is a vectorized version of a for loop for all frequencies.
    # # Creating Identity matrices of shape (nports,nports) for each nfreqs
    Id = jnp.eye(nports, dtype=complex)[None, :, :]  # (1, nports, nports)
    Id = jnp.broadcast_to(Id, (nfreqs, nports, nports))

    if s_def == 'power':
        # Power-waves. Eq.(19) from [Kurokawa et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs

        F, G = jnp.zeros_like(s), jnp.zeros_like(s)
        diag_idx = jnp.arange(F.shape[1])
        F = F.at[:, diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0.real)))
        G = G.at[:, diag_idx, diag_idx].set(z0)        
        # z = jnp.linalg.solve(nudge_eig((Id - s) @ F), (s @ G + jnp.conjugate(G)) @ F)
        z = jnp.linalg.solve(nudge_diag((Id - s) @ F), (s @ G + jnp.conjugate(G)) @ F)
    elif s_def == 'traveling':
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating diagonal matrices of shape (nports, nports) for each nfreqs
        sqrtz0 = jnp.zeros_like(s)
        diag_idx = jnp.arange(s.shape[1])
        sqrtz0 = sqrtz0.at[:, diag_idx, diag_idx].set(jnp.sqrt(z0))
        z = sqrtz0 @ jnp.linalg.solve(nudge_diag(Id - s), (Id + s) @ sqrtz0)        
    else:
        raise ValueError(f'Unknown s_def: {s_def}')

    return z

def z2s(z: NumberLike, z0:NumberLike = 50, s_def = 'power') -> jnp.ndarray:
    """
    Convert Impedance (Z) parameters to S-parameters.

    Parameters
    ----------
    z : jnp.ndarray
        The Impedance matrix with shape `(nfreqs, nports, nports)`.
    z0 : NumberLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The S-parameter matrix with shape `(nfreqs, nports, nports)`.
    """
    nfreqs, nports, nports = z.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    z0 = z0.astype(dtype=complex)
    z0 = jnp.where(z0.real == 0, z0 + ZERO, z0)
    z = jnp.array(z, dtype=complex)

    if s_def == 'power':
        # Power-waves. Eq.(18) from [Kurokawa et al.3]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F, G = jnp.zeros_like(z), jnp.zeros_like(z)
        diag_idx = jnp.arange(F.shape[1])
        F = F.at[:, diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0.real)))
        G = G.at[:, diag_idx, diag_idx].set(z0)
        s = rsolve(F @ (z + G), F @ (z - jnp.conjugate(G)))
    elif s_def == 'traveling':
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating Identity matrices of shape (nports,nports) for each nfreqs
        Id, sqrty0 = jnp.zeros_like(z), jnp.zeros_like(z) # (nfreqs, nports, nports)
        diag_idx = jnp.arange(z.shape[1])
        Id = Id.at[:, diag_idx, diag_idx].set(1.0)
        sqrty0 = sqrty0.at[:, diag_idx, diag_idx].set(jnp.sqrt(1.0/z0))
        s = rsolve(sqrty0 @ z @ sqrty0 + Id, sqrty0 @ z @ sqrty0 - Id)        
    else:
        raise ValueError(f'Unknown s_def: {s_def}')

    return s

def renormalize_s(s: jnp.ndarray, z_old: NumberLike, z_new: NumberLike, s_def_old='power', s_def_new='power', method='hub') -> jnp.ndarray:
    """
    Renormalize S-parameters from one impedance/definition to another.

    Parameters
    ----------
    s : jnp.ndarray
        The input S-parameter matrix.
    z_old : NumberLike
        The original characteristic impedance.
    z_new : NumberLike
        The new characteristic impedance.
    s_def_old : str, optional, default='power'
        The original S-parameter definition.
    s_def_new : str, optional, default='power'
        The new S-parameter definition.
    method: str, optional, default='direct'
        The algorithm to use. Can be 'direct' or 'hub'.

    Returns
    -------
    jnp.ndarray
        The renormalized S-parameter matrix.
    """
    if method == 'hub':
        return z2s(s2z(s, z0=z_old, s_def=s_def_old), z0=z_new, s_def=s_def_new)
    elif method == 'mobius':
        s_renorm = renormalize_s_mobius(s, z_old, z_new, s_def=s_def_old)
        s_redef = s2s(s_renorm, z_new, s_def_new=s_def_new, s_def_old=s_def_old)
        return s_redef
    else:
        raise ValueError("Unknown S renormalization method")

def renormalize_s_mobius(
    s: jnp.ndarray,
    z_old: NumberLike,
    z_new: NumberLike,
    s_def: str = 'power',
) -> jnp.ndarray:
    if s_def != 'traveling':
        s = s2s(s, z_old, 'traveling', s_def)

    nfreqs, nports, _ = s.shape

    Z_A = fix_z0_shape(z_old, nfreqs, nports)
    Z_B = fix_z0_shape(z_new, nfreqs, nports)

    I = jnp.eye(nports, dtype=s.dtype)

    def renorm_per_freq(s_f, z_a, z_b):

        gamma = (z_b - z_a) / (z_b + z_a)

        GammaS = gamma[:, None] * s_f
        S_minus_G = s_f - jnp.diag(gamma)

        A = I - GammaS
        B = S_minus_G

        # Solve X A = B
        return jnp.linalg.solve(A.T.conj(), B.T.conj()).T.conj()

    s_renorm = jax.vmap(renorm_per_freq)(s, Z_A, Z_B)

    if s_def != 'traveling':
        s_renorm = s2s(s_renorm, z_old, s_def, 'traveling')

    return s_renorm