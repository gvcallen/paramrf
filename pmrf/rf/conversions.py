"""
RF parameter conversion algorithms.
"""

import jax.numpy as jnp
import jax
from jaxtyping import ArrayLike

from pmrf.math import rsolve, nudge_diag
from pmrf.utils.rf import fix_z0_shape
from pmrf.rf.mna import MNAStamp

ZERO = 1e-4

def s2s(s: ArrayLike, z0: ArrayLike, s_def_new: str, s_def_old: str):
    """
    Convert S-parameters between different definitions (e.g., Power waves vs Traveling waves).

    This function handles the conversion logic defined by `s_def_old` to `s_def_new`.
    It supports complex characteristic impedances and accepts both single-frequency
    (2D) and multi-frequency (3D) S-parameter matrices.

    Parameters
    ----------
    s : ArrayLike
        The S-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z0 : ArrayLike
        The characteristic impedance. Can be a scalar, or an array broadcastable
        to `(nports,)` or `(nfreqs, nports)`.
    s_def_new : str
        The target S-parameter definition. Options: 'power', 'traveling'.
    s_def_old : str
        The source S-parameter definition. Options: 'power', 'traveling'.

    Returns
    -------
    jnp.ndarray
        The converted S-parameter matrix matching the shape of the input `s`.
    """
    if s_def_new == s_def_old:
        return s

    s_arr = jnp.asarray(s)
    
    if s_arr.ndim == 3:
        nfreqs, nports, _ = s_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(s2s, in_axes=(0, 0, None, None))(s_arr, z0_fixed, s_def_new, s_def_old)
        
    elif s_arr.ndim == 2:
        nports = s_arr.shape[0]
        z0_arr = fix_z0_shape(z0, 1, nports)[0]

        all_real = jnp.isreal(z0_arr).all()
        
        def real_branch():
            return s_arr
        
        def imag_branch():
            # Calculate port voltages and currents using the old s_def.
            F, G = jnp.zeros_like(s_arr), jnp.zeros_like(s_arr)
            diag_idx = jnp.arange(nports)
            
            if s_def_old == 'power':
                F = F.at[diag_idx, diag_idx].set(1.0 / (jnp.sqrt(z0_arr.real)))
                G = G.at[diag_idx, diag_idx].set(z0_arr)        
                Id = jnp.eye(nports, dtype=complex)
                v = F @ (G.conjugate() + G @ s_arr)
                i = F @ (Id - s_arr)
            elif s_def_old == 'traveling':
                F = F.at[diag_idx, diag_idx].set(jnp.sqrt(z0_arr))
                G = G.at[diag_idx, diag_idx].set(1.0 / (jnp.sqrt(z0_arr)))        
                Id = jnp.eye(nports, dtype=complex)
                v = F @ (Id + s_arr)
                i = G @ (Id - s_arr)
            else:
                raise ValueError(f'Unknown s_def: {s_def_old}')

            # Calculate a and b waves from the voltages and currents.
            F, G = jnp.zeros_like(s_arr), jnp.zeros_like(s_arr)
            if s_def_new == 'power':
                F = F.at[diag_idx, diag_idx].set(1.0 / (2.0 * jnp.sqrt(z0_arr.real)))
                G = G.at[diag_idx, diag_idx].set(z0_arr)    
                a = F @ (v + G @ i)
                b = F @ (v - G.conjugate() @ i)
            elif s_def_new == 'traveling':
                F = F.at[diag_idx, diag_idx].set(1.0 / (jnp.sqrt(z0_arr)))
                G = G.at[diag_idx, diag_idx].set(z0_arr) 
                a = F @ (v + G @ i)
                b = F @ (v - G @ i)
            else:
                raise ValueError(f'Unknown s_def: {s_def_new}')

            # New S-parameter matrix from a and b waves.
            s_new = jnp.zeros_like(s_arr)
            for n in range(nports):
                for m in range(nports):
                    s_new = s_new.at[m, n].set(b[m, n] / a[n, n])

            return s_new

        return jax.lax.cond(all_real, real_branch, imag_branch)
        
    else:
        raise ValueError(f"S-parameters must be 2D (nports, nports) or 3D (nfreqs, nports, nports). Got {s_arr.ndim}D.")

def a2s(a: jnp.ndarray, z0: ArrayLike = 50) -> jnp.ndarray:
    """
    Convert ABCD parameters to S-parameters.

    Parameters
    ----------
    a : jnp.ndarray
        The ABCD parameter matrix with shape `(2, 2)` or `(nfreqs, 2, 2)`.
    z0 : ArrayLike, optional, default=50
        The characteristic impedance.

    Returns
    -------
    jnp.ndarray
        The S-parameter matrix with shape matching the input `a`.

    Raises
    ------
    IndexError
        If the input is not a 2-port network.
    """
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    a_arr = jnp.asarray(a)
    
    if a_arr.ndim == 3:
        nfreqs, nports, _ = a_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(a2s, in_axes=(0, 0))(a_arr, z0_fixed)
        
    elif a_arr.ndim == 2:
        nports = a_arr.shape[0]
        if nports != 2:
            raise IndexError('abcd parameters are defined for 2-ports networks only')

        z0_arr = fix_z0_shape(z0, 1, nports)[0]
        z01 = z0_arr[0]
        z02 = z0_arr[1]
        A = a_arr[0,0]
        B = a_arr[0,1]
        C = a_arr[1,0]
        D = a_arr[1,1]
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
        
    else:
        raise ValueError(f"ABCD parameters must be 2D or 3D. Got {a_arr.ndim}D.")

def s2a(s: jnp.ndarray, z0: ArrayLike = 50) -> jnp.ndarray:
    """
    Convert S-parameters to ABCD parameters.

    Parameters
    ----------
    s : jnp.ndarray
        The S-parameter matrix with shape `(2, 2)` or `(nfreqs, 2, 2)`.
    z0 : ArrayLike, optional, default=50
        The characteristic impedance.

    Returns
    -------
    jnp.ndarray
        The ABCD parameter matrix with shape matching the input `s`.

    Raises
    ------
    IndexError
        If the input is not a 2-port network.
    """
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    s_arr = jnp.asarray(s)
    
    if s_arr.ndim == 3:
        nfreqs, nports, _ = s_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(s2a, in_axes=(0, 0))(s_arr, z0_fixed)
        
    elif s_arr.ndim == 2:
        nports = s_arr.shape[0]
        if nports != 2:
            raise IndexError('abcd parameters are defined for 2-ports networks only')

        z0_arr = fix_z0_shape(z0, 1, nports)[0]
        z01 = z0_arr[0]
        z02 = z0_arr[1]
        denom = (2*s_arr[1,0]*jnp.sqrt(z01.real * z02.real))
        
        a = jnp.array([
            [
                ((z01.conj() + s_arr[0,0]*z01)*(1 - s_arr[1,1]) + s_arr[0,1]*s_arr[1,0]*z01) / denom,
                ((1 - s_arr[0,0])*(1 - s_arr[1,1]) - s_arr[0,1]*s_arr[1,0]) / denom,
            ],
            [
                ((z01.conj() + s_arr[0,0]*z01)*(z02.conj() + s_arr[1,1]*z02) - s_arr[0,1]*s_arr[1,0]*z01*z02) / denom,
                ((1 - s_arr[0,0])*(z02.conj() + s_arr[1,1]*z02) + s_arr[0,1]*s_arr[1,0]*z02) / denom,
            ],
        ]).transpose()
        return a
        
    else:
        raise ValueError(f"S-parameters must be 2D or 3D. Got {s_arr.ndim}D.")

def s2y(s: jnp.ndarray, z0: ArrayLike = 50, s_def: str = 'power') -> jnp.ndarray:
    """
    Convert S-parameters to Admittance (Y) parameters.

    Parameters
    ----------
    s : jnp.ndarray
        The S-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z0 : ArrayLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The Admittance matrix with shape matching the input `s`.
    """
    s_arr = jnp.asarray(s)
    
    if s_arr.ndim == 3:
        nfreqs, nports, _ = s_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(s2y, in_axes=(0, 0, None))(s_arr, z0_fixed, s_def)
        
    elif s_arr.ndim == 2:
        nports = s_arr.shape[0]
        z0_arr = fix_z0_shape(z0, 1, nports)[0]
        z0_arr = z0_arr.astype(dtype=complex)
        z0_arr = jnp.where(z0_arr.real == 0, z0_arr + ZERO, z0_arr)

        s_arr = jnp.array(s_arr, dtype=complex)

        # Creating Identity matrices of shape (nports,nports) for each nfreqs
        Id = jnp.eye(nports, dtype=complex)

        if s_def == 'power':
            F, F_inv, G = jnp.zeros_like(s_arr), jnp.zeros_like(s_arr), jnp.zeros_like(s_arr)
            diag_idx = jnp.arange(nports)
            
            # F_inv is the inverse of F: a diagonal matrix of 2 * sqrt(Re(Z0))
            F = F.at[diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0_arr.real)))
            F_inv = F_inv.at[diag_idx, diag_idx].set(2 * jnp.sqrt(z0_arr.real))
            G = G.at[diag_idx, diag_idx].set(z0_arr)
            
            # Left-solve: X = A^-1 B  =>  jnp.linalg.solve(A, B)
            # Y = F_inv @ (S @ G + G^*)^-1 @ (I - S) @ F
            A = s_arr @ G + jnp.conjugate(G)
            B = Id - s_arr
            
            y = F_inv @ jnp.linalg.solve(nudge_diag(A), B) @ F

        elif s_def == 'traveling':
            # Creating diagonal matrices of 1 / sqrt(Z0)
            inv_sqrtz0 = jnp.zeros_like(s_arr)
            diag_idx = jnp.arange(nports)
            inv_sqrtz0 = inv_sqrtz0.at[diag_idx, diag_idx].set(1.0 / jnp.sqrt(z0_arr))
            
            # Y = Z0^-1/2 @ (I + S)^-1 @ (I - S) @ Z0^-1/2
            A = Id + s_arr
            B = Id - s_arr
            
            y = inv_sqrtz0 @ jnp.linalg.solve(nudge_diag(A), B) @ inv_sqrtz0

        else:
            raise ValueError(f"Unknown s_def: {s_def}")

        return y
        
    else:
        raise ValueError(f"S-parameters must be 2D or 3D. Got {s_arr.ndim}D.")

def y2s(y: jnp.ndarray, z0: ArrayLike = 50, s_def = 'power') -> jnp.ndarray:
    """
    Convert Admittance (Y) parameters to S-parameters.

    Parameters
    ----------
    y : jnp.ndarray
        The Admittance matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z0 : ArrayLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The S-parameter matrix with shape matching the input `y`.
    """
    y_arr = jnp.asarray(y)
    
    if y_arr.ndim == 3:
        nfreqs, nports, _ = y_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(y2s, in_axes=(0, 0, None))(y_arr, z0_fixed, s_def)
        
    elif y_arr.ndim == 2:
        nports = y_arr.shape[0]
        z0_arr = fix_z0_shape(z0, 1, nports)[0]
        z0_arr = z0_arr.astype(dtype=complex)
        z0_arr = jnp.where(z0_arr.real == 0, z0_arr + ZERO, z0_arr)

        y_arr = jnp.array(y_arr, dtype=complex)

        # The following is a vectorized version of a for loop for all frequencies.
        # Creating Identity matrices of shape (nports,nports) for each nfreqs
        Id = jnp.eye(nports, dtype=complex)

        if s_def == 'power':
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs
            F, G = jnp.zeros_like(y_arr), jnp.zeros_like(y_arr)
            diag_idx = jnp.arange(nports)
            F = F.at[diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0_arr.real)))
            G = G.at[diag_idx, diag_idx].set(z0_arr)        
            s = rsolve(F @ (Id + G @ y_arr), F @ (Id - jnp.conjugate(G) @ y_arr))
        elif s_def == 'traveling':
            # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
            # Creating diagonal matrices of shape (nports, nports) for each nfreqs
            sqrtz0 = jnp.zeros_like(y_arr)
            diag_idx = jnp.arange(nports)
            sqrtz0 = sqrtz0.at[diag_idx, diag_idx].set(jnp.sqrt(z0_arr))
            s = rsolve(Id + sqrtz0 @ y_arr @ sqrtz0, Id - sqrtz0 @ y_arr @ sqrtz0)
        else:
            raise ValueError(f'Unknown s_def: {s_def}')

        return s
        
    else:
        raise ValueError(f"Y-parameters must be 2D or 3D. Got {y_arr.ndim}D.")

def s2z(s: jnp.ndarray, z0: ArrayLike = 50, s_def = 'power') -> jnp.ndarray:
    """
    Convert S-parameters to Impedance (Z) parameters.

    Parameters
    ----------
    s : jnp.ndarray
        The S-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z0 : ArrayLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The Impedance matrix with shape matching the input `s`.
    """
    s_arr = jnp.asarray(s)
    
    if s_arr.ndim == 3:
        nfreqs, nports, _ = s_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(s2z, in_axes=(0, 0, None))(s_arr, z0_fixed, s_def)
        
    elif s_arr.ndim == 2:
        nports = s_arr.shape[0]
        z0_arr = fix_z0_shape(z0, 1, nports)[0]
        z0_arr = z0_arr.astype(dtype=complex)
        z0_arr = jnp.where(z0_arr.real == 0, z0_arr + ZERO, z0_arr)

        s_arr = jnp.array(s_arr, dtype=complex)

        # The following is a vectorized version of a for loop for all frequencies.
        # # Creating Identity matrices of shape (nports,nports) for each nfreqs
        Id = jnp.eye(nports, dtype=complex)

        if s_def == 'power':
            # Power-waves. Eq.(19) from [Kurokawa et al.]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs

            F, G = jnp.zeros_like(s_arr), jnp.zeros_like(s_arr)
            diag_idx = jnp.arange(nports)
            F = F.at[diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0_arr.real)))
            G = G.at[diag_idx, diag_idx].set(z0_arr)        
            z = jnp.linalg.solve(nudge_diag((Id - s_arr) @ F), (s_arr @ G + jnp.conjugate(G)) @ F)
        elif s_def == 'traveling':
            # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
            # Creating diagonal matrices of shape (nports, nports) for each nfreqs
            sqrtz0 = jnp.zeros_like(s_arr)
            diag_idx = jnp.arange(nports)
            sqrtz0 = sqrtz0.at[diag_idx, diag_idx].set(jnp.sqrt(z0_arr))
            z = sqrtz0 @ jnp.linalg.solve(nudge_diag(Id - s_arr), (Id + s_arr) @ sqrtz0)        
        else:
            raise ValueError(f'Unknown s_def: {s_def}')

        return z
        
    else:
        raise ValueError(f"S-parameters must be 2D or 3D. Got {s_arr.ndim}D.")

def z2s(z: ArrayLike, z0:ArrayLike = 50, s_def = 'power') -> jnp.ndarray:
    """
    Convert Impedance (Z) parameters to S-parameters.

    Parameters
    ----------
    z : jnp.ndarray
        The Impedance matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z0 : ArrayLike, optional, default=50
        The characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition ('power' or 'traveling').

    Returns
    -------
    jnp.ndarray
        The S-parameter matrix with shape matching the input `z`.
    """
    z_arr = jnp.asarray(z)
    
    if z_arr.ndim == 3:
        nfreqs, nports, _ = z_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(z2s, in_axes=(0, 0, None))(z_arr, z0_fixed, s_def)
        
    elif z_arr.ndim == 2:
        nports = z_arr.shape[0]
        z0_arr = fix_z0_shape(z0, 1, nports)[0]
        z0_arr = z0_arr.astype(dtype=complex)
        z0_arr = jnp.where(z0_arr.real == 0, z0_arr + ZERO, z0_arr)
        z_arr = jnp.array(z_arr, dtype=complex)

        if s_def == 'power':
            # Power-waves. Eq.(18) from [Kurokawa et al.3]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs
            F, G = jnp.zeros_like(z_arr), jnp.zeros_like(z_arr)
            diag_idx = jnp.arange(nports)
            F = F.at[diag_idx, diag_idx].set(1.0 / (2 * jnp.sqrt(z0_arr.real)))
            G = G.at[diag_idx, diag_idx].set(z0_arr)
            s = rsolve(F @ (z_arr + G), F @ (z_arr - jnp.conjugate(G)))
        elif s_def == 'traveling':
            # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
            # Creating Identity matrices of shape (nports,nports) for each nfreqs
            Id, sqrty0 = jnp.zeros_like(z_arr), jnp.zeros_like(z_arr)
            diag_idx = jnp.arange(nports)
            Id = Id.at[diag_idx, diag_idx].set(1.0)
            sqrty0 = sqrty0.at[diag_idx, diag_idx].set(jnp.sqrt(1.0/z0_arr))
            s = rsolve(sqrty0 @ z_arr @ sqrty0 + Id, sqrty0 @ z_arr @ sqrty0 - Id)        
        else:
            raise ValueError(f'Unknown s_def: {s_def}')

        return s
        
    else:
        raise ValueError(f"Z-parameters must be 2D or 3D. Got {z_arr.ndim}D.")

def y2z(y: jnp.ndarray) -> jnp.ndarray:
    """
    Directly convert Admittance (Y) parameters to Impedance (Z) parameters.

    Parameters
    ----------
    y : jnp.ndarray
        The Admittance matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.

    Returns
    -------
    jnp.ndarray
        The Impedance matrix with shape matching the input `y`.
    """
    y_arr = jnp.asarray(y)
    
    if y_arr.ndim == 3:
        return jax.vmap(y2z)(y_arr)
    elif y_arr.ndim == 2:
        return jnp.linalg.inv(nudge_diag(y_arr))
    else:
        raise ValueError(f"Y-parameters must be 2D or 3D. Got {y_arr.ndim}D.")

def z2y(z: jnp.ndarray) -> jnp.ndarray:
    """
    Directly convert Impedance (Z) parameters to Admittance (Y) parameters.

    Parameters
    ----------
    z : jnp.ndarray
        The Impedance matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.

    Returns
    -------
    jnp.ndarray
        The Admittance matrix with shape matching the input `z`.
    """
    z_arr = jnp.asarray(z)
    
    if z_arr.ndim == 3:
        return jax.vmap(z2y)(z_arr)
    elif z_arr.ndim == 2:
        return jnp.linalg.inv(nudge_diag(z_arr))
    else:
        raise ValueError(f"Z-parameters must be 2D or 3D. Got {z_arr.ndim}D.")

def a2y(a: jnp.ndarray) -> jnp.ndarray:
    """
    Directly convert ABCD parameters to Admittance (Y) parameters.

    Parameters
    ----------
    a : jnp.ndarray
        The ABCD parameter matrix with shape `(2, 2)` or `(nfreqs, 2, 2)`.

    Returns
    -------
    jnp.ndarray
        The Admittance matrix with shape matching the input `a`.
    """
    a_arr = jnp.asarray(a)
    
    if a_arr.ndim == 3:
        return jax.vmap(a2y)(a_arr)
    elif a_arr.ndim == 2:
        nports = a_arr.shape[0]
        if nports != 2:
            raise IndexError('ABCD parameters are defined for 2-port networks only')

        A = a_arr[0, 0]
        B = a_arr[0, 1]
        C = a_arr[1, 0]
        D = a_arr[1, 1]

        denom = jnp.where(B == 0, 1e-15, B)

        y = jnp.array([
            [ D / denom,               -(A * D - B * C) / denom ],
            [ -1.0 / denom,            A / denom                ],
        ])
        
        return y
    else:
        raise ValueError(f"ABCD parameters must be 2D or 3D. Got {a_arr.ndim}D.")

def y2a(y: jnp.ndarray) -> jnp.ndarray:
    """
    Directly convert Admittance (Y) parameters to ABCD parameters.

    Parameters
    ----------
    y : jnp.ndarray
        The Admittance matrix with shape `(2, 2)` or `(nfreqs, 2, 2)`.

    Returns
    -------
    jnp.ndarray
        The ABCD parameter matrix with shape matching the input `y`.
    """
    y_arr = jnp.asarray(y)
    
    if y_arr.ndim == 3:
        return jax.vmap(y2a)(y_arr)
    elif y_arr.ndim == 2:
        nports = y_arr.shape[0]
        if nports != 2:
            raise IndexError('ABCD parameters are defined for 2-port networks only')

        y11 = y_arr[0, 0]
        y12 = y_arr[0, 1]
        y21 = y_arr[1, 0]
        y22 = y_arr[1, 1]

        denom = jnp.where(y21 == 0, 1e-15, y21)
        delta_y = y11 * y22 - y12 * y21

        a = jnp.array([
            [ -y22 / denom,            -1.0 / denom ],
            [ -delta_y / denom,        -y11 / denom ],
        ])

        return a
    else:
        raise ValueError(f"Y-parameters must be 2D or 3D. Got {y_arr.ndim}D.")

def a2z(a: jnp.ndarray) -> jnp.ndarray:
    """
    Directly convert ABCD parameters to Impedance (Z) parameters.

    Parameters
    ----------
    a : jnp.ndarray
        The ABCD parameter matrix with shape `(2, 2)` or `(nfreqs, 2, 2)`.

    Returns
    -------
    jnp.ndarray
        The Impedance matrix with shape matching the input `a`.
    """
    a_arr = jnp.asarray(a)
    
    if a_arr.ndim == 3:
        return jax.vmap(a2z)(a_arr)
    elif a_arr.ndim == 2:
        nports = a_arr.shape[0]
        if nports != 2:
            raise IndexError('ABCD parameters are defined for 2-port networks only')

        A = a_arr[0, 0]
        B = a_arr[0, 1]
        C = a_arr[1, 0]
        D = a_arr[1, 1]

        denom = jnp.where(C == 0, 1e-15, C)

        z = jnp.array([
            [ A / denom,               (A * D - B * C) / denom ],
            [ 1.0 / denom,             D / denom               ],
        ])

        return z
    else:
        raise ValueError(f"ABCD parameters must be 2D or 3D. Got {a_arr.ndim}D.")

def z2a(z: jnp.ndarray) -> jnp.ndarray:
    """
    Directly convert Impedance (Z) parameters to ABCD parameters.

    Parameters
    ----------
    z : jnp.ndarray
        The Impedance matrix with shape `(2, 2)` or `(nfreqs, 2, 2)`.

    Returns
    -------
    jnp.ndarray
        The ABCD parameter matrix with shape matching the input `z`.
    """
    z_arr = jnp.asarray(z)
    
    if z_arr.ndim == 3:
        return jax.vmap(z2a)(z_arr)
    elif z_arr.ndim == 2:
        nports = z_arr.shape[0]
        if nports != 2:
            raise IndexError('ABCD parameters are defined for 2-port networks only')

        z11 = z_arr[0, 0]
        z12 = z_arr[0, 1]
        z21 = z_arr[1, 0]
        z22 = z_arr[1, 1]

        denom = jnp.where(z21 == 0, 1e-15, z21)
        delta_z = z11 * z22 - z12 * z21

        a = jnp.array([
            [ z11 / denom,             delta_z / denom ],
            [ 1.0 / denom,             z22 / denom     ],
        ])

        return a
    else:
        raise ValueError(f"Z-parameters must be 2D or 3D. Got {z_arr.ndim}D.")

def y2mna(y: ArrayLike) -> MNAStamp:
    """
    Convert Y-parameters to a Modified Nodal Analysis (MNA) stamp.
    
    Since Y-parameters already represent a pure nodal formulation, 
    this function generates a standard stamp with zero auxiliary variables.
    
    Parameters
    ----------
    y : ArrayLike
        The Y-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
        
    Returns
    -------
    MNAStamp
        The MNA representation containing Y, B, C, and D matrices.
    """
    y_arr = jnp.asarray(y)
    
    if y_arr.ndim == 3:
        return jax.vmap(y2mna)(y_arr)
    elif y_arr.ndim == 2:
        nports = y_arr.shape[-1]
        
        # For pure Y-parameters, the number of auxiliary variables (K) is 0.
        # JAX handles 0-dimension sizes cleanly.
        B = jnp.zeros((nports, 0), dtype=y_arr.dtype)
        C = jnp.zeros((0, nports), dtype=y_arr.dtype)
        D = jnp.zeros((0, 0), dtype=y_arr.dtype)
        
        return MNAStamp(Y=y_arr, B=B, C=C, D=D)
    else:
        raise ValueError(f"Y-parameters must be 2D or 3D. Got {y_arr.ndim}D.")

def z2mna(z: ArrayLike) -> MNAStamp:
    """
    Convert Impedance (Z) parameters to a Modified Nodal Analysis (MNA) stamp.
    
    This formulation maps the port currents as the auxiliary variables, allowing 
    raw Z-parameters to be stamped directly into the MNA system without requiring 
    a potentially singular Z-to-Y matrix inversion.
    
    Parameters
    ----------
    z : ArrayLike
        The Z-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
        
    Returns
    -------
    MNAStamp
        The MNA representation containing Y, B, C, and D matrices.
    """
    z_arr = jnp.asarray(z)
    
    if z_arr.ndim == 3:
        return jax.vmap(z2mna)(z_arr)
    elif z_arr.ndim == 2:
        nports = z_arr.shape[-1]
        
        I = jnp.eye(nports, dtype=z_arr.dtype)
        
        # Direct Z-parameter MNA derivation
        Y = jnp.zeros_like(z_arr)
        B = I
        C = I
        D = -z_arr
        
        return MNAStamp(Y=Y, B=B, C=C, D=D)
    else:
        raise ValueError(f"Z-parameters must be 2D or 3D. Got {z_arr.ndim}D.")

def a2mna(a: ArrayLike) -> MNAStamp:
    """
    Convert Transfer (ABCD) parameters to a Modified Nodal Analysis (MNA) stamp.
    
    This formulation maps the port currents as the auxiliary variables, 
    avoiding potentially singular matrix inversions.
    
    Parameters
    ----------
    a : ArrayLike
        The ABCD parameter matrix with shape `(2, 2)` or `(nfreqs, 2, 2)`.
        
    Returns
    -------
    MNAStamp
        The MNA representation containing Y, B, C, and D matrices.
        
    Raises
    ------
    IndexError
        If the input is not a 2-port network.
    """
    a_arr = jnp.asarray(a)
    
    if a_arr.ndim == 3:
        return jax.vmap(a2mna)(a_arr)
    elif a_arr.ndim == 2:
        nports = a_arr.shape[0]
        if nports != 2:
            raise IndexError('ABCD parameters are defined for 2-port networks only')
            
        A = a_arr[0, 0]
        B = a_arr[0, 1]
        C = a_arr[1, 0]
        D = a_arr[1, 1]

        Y = jnp.zeros_like(a_arr)
        I = jnp.eye(2, dtype=a_arr.dtype)
        B_mat = I

        # Assemble C matrix: [[1, -A], [0, -C]]
        C_mat = jnp.zeros_like(a_arr)
        C_mat = C_mat.at[0, 0].set(1.0)
        C_mat = C_mat.at[0, 1].set(-A)
        C_mat = C_mat.at[1, 0].set(0.0)
        C_mat = C_mat.at[1, 1].set(-C)

        # Assemble D matrix: [[0, B], [1, D]]
        D_mat = jnp.zeros_like(a_arr)
        D_mat = D_mat.at[0, 0].set(0.0)
        D_mat = D_mat.at[0, 1].set(B)
        D_mat = D_mat.at[1, 0].set(1.0)
        D_mat = D_mat.at[1, 1].set(D)

        return MNAStamp(Y=Y, B=B_mat, C=C_mat, D=D_mat)
    else:
        raise ValueError(f"ABCD parameters must be 2D or 3D. Got {a_arr.ndim}D.")

def s2mna(s: ArrayLike, z0: ArrayLike) -> MNAStamp:
    """
    Convert S-parameters to a Modified Nodal Analysis (MNA) stamp.
    
    This formulation maps the incident voltage waves as the auxiliary 
    variables, allowing raw S-parameters to be stamped directly into 
    the MNA system without any matrix inversions.
    
    Parameters
    ----------
    s : ArrayLike
        The S-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z0 : ArrayLike
        The characteristic impedance. Can be a scalar or an array broadcastable
        to `(nports,)` or `(nfreqs, nports)`.
        
    Returns
    -------
    MNAStamp
        The MNA representation containing Y, B, C, and D matrices.
    """
    s_arr = jnp.asarray(s)
    
    if s_arr.ndim == 3:
        nfreqs, nports, _ = s_arr.shape
        z0_fixed = fix_z0_shape(z0, nfreqs, nports)
        return jax.vmap(s2mna, in_axes=(0, 0))(s_arr, z0_fixed)
    elif s_arr.ndim == 2:
        nports = s_arr.shape[0]
        z0_arr = fix_z0_shape(z0, 1, nports)[0]
        
        # Calculate Y0 and expand dims to broadcast across the columns of (I - S)
        y0 = 1.0 / z0_arr
        y0 = jnp.expand_dims(y0, axis=-1) 
        
        I = jnp.eye(nports, dtype=s_arr.dtype)
        
        # Universal S-parameter MNA derivation
        Y = jnp.zeros_like(s_arr)
        B = y0 * (I - s_arr)
        C = I
        D = -(I + s_arr)
        
        return MNAStamp(Y=Y, B=B, C=C, D=D)
    else:
        raise ValueError(f"S-parameters must be 2D or 3D. Got {s_arr.ndim}D.")

def renormalize_s(s: jnp.ndarray, z_old: ArrayLike, z_new: ArrayLike, s_def_old='power', s_def_new='power', method='mobius') -> jnp.ndarray:
    """
    Renormalize S-parameters from one impedance/definition to another.
    
    Includes branches to skip normalization if not needed.

    Parameters
    ----------
    s : jnp.ndarray
        The input S-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z_old : ArrayLike
        The original characteristic impedance.
    z_new : ArrayLike
        The new characteristic impedance.
    s_def_old : str, optional, default='power'
        The original S-parameter definition.
    s_def_new : str, optional, default='power'
        The new S-parameter definition.
    method: str, optional, default='mobius'
        The algorithm to use. Can be 'mobius' or 'hub'.

    Returns
    -------
    jnp.ndarray
        The renormalized S-parameter matrix.
    """
    s_arr = jnp.asarray(s)
    
    if s_arr.ndim == 3:
        nfreqs, nports, _ = s_arr.shape
        z_old_fixed = fix_z0_shape(z_old, nfreqs, nports)
        z_new_fixed = fix_z0_shape(z_new, nfreqs, nports)
        return jax.vmap(renormalize_s, in_axes=(0, 0, 0, None, None, None))(s_arr, z_old_fixed, z_new_fixed, s_def_old, s_def_new, method)
        
    elif s_arr.ndim == 2:
        nports = s_arr.shape[0]
        z_old_arr = fix_z0_shape(z_old, 1, nports)[0]
        z_new_arr = fix_z0_shape(z_new, 1, nports)[0]
        
        defs_match = (s_def_old == s_def_new)
        z_match = jnp.all(z_old_arr == z_new_arr)
        is_matched = jnp.logical_and(defs_match, z_match)
        
        def _do_renorm():
            if method == 'hub':
                return z2s(s2z(s_arr, z0=z_old_arr, s_def=s_def_old), z0=z_new_arr, s_def=s_def_new)
            elif method == 'mobius':
                s_renorm = renormalize_s_mobius(s_arr, z_old_arr, z_new_arr, s_def=s_def_old)
                s_redef = s2s(s_renorm, z_new_arr, s_def_new=s_def_new, s_def_old=s_def_old)
                return s_redef
            else:
                raise ValueError(f"Unknown S renormalization method: {method}")
                
        def _identity():
            return s_arr
            
        return jax.lax.cond(is_matched, _identity, _do_renorm)
        
    else:
        raise ValueError(f"S-parameters must be 2D or 3D. Got {s_arr.ndim}D.")

def renormalize_s_mobius(
    s: jnp.ndarray,
    z_old: ArrayLike,
    z_new: ArrayLike,
    s_def: str = 'power',
) -> jnp.ndarray:
    """
    Renormalize S-parameters using the mobius transform.
    
    Parameters
    ----------
    s : jnp.ndarray
        The input S-parameter matrix with shape `(nports, nports)` or `(nfreqs, nports, nports)`.
    z_old : ArrayLike
        The original characteristic impedance.
    z_new : ArrayLike
        The new characteristic impedance.
    s_def : str, optional, default='power'
        The S-parameter definition.
        
    Returns
    -------
    jnp.ndarray
        The renormalized S-parameter matrix.
    """
    s_arr = jnp.asarray(s)
    
    if s_arr.ndim == 3:
        nfreqs, nports, _ = s_arr.shape
        z_old_fixed = fix_z0_shape(z_old, nfreqs, nports)
        z_new_fixed = fix_z0_shape(z_new, nfreqs, nports)
        return jax.vmap(renormalize_s_mobius, in_axes=(0, 0, 0, None))(s_arr, z_old_fixed, z_new_fixed, s_def)
        
    elif s_arr.ndim == 2:
        nports = s_arr.shape[0]
        z_old_arr = fix_z0_shape(z_old, 1, nports)[0]
        z_new_arr = fix_z0_shape(z_new, 1, nports)[0]

        if s_def != 'traveling':
            s_arr = s2s(s_arr, z_old_arr, 'traveling', s_def)

        I = jnp.eye(nports, dtype=s_arr.dtype)

        gamma = (z_new_arr - z_old_arr) / (z_new_arr + z_old_arr)

        GammaS = gamma[:, None] * s_arr
        S_minus_G = s_arr - jnp.diag(gamma)

        # 1. Add numerical stabilization to prevent singular matrix inversion
        A = nudge_diag(I - GammaS) 
        B = S_minus_G

        # Solve X A = B => X = (S - Gamma) @ (I - Gamma S)^-1
        X = jnp.linalg.solve(A.T.conj(), B.T.conj()).T.conj()

        # 2. Apply the wave-amplitude scaling matrix M
        M = (z_new_arr + z_old_arr) / (2 * jnp.sqrt(z_old_arr * z_new_arr))
        
        # M @ X @ M^-1 is mathematically equivalent to multiplying each X_ij by (M_i / M_j)
        s_renorm = X * (M[:, None] / M[None, :])

        if s_def != 'traveling':
            s_renorm = s2s(s_renorm, z_old_arr, s_def, 'traveling')

        return s_renorm
        
    else:
        raise ValueError(f"S-parameters must be 2D or 3D. Got {s_arr.ndim}D.")