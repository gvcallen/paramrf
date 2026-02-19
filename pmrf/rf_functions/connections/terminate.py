import jax.numpy as jnp
from pmrf.rf_functions.conversions import fix_z0_shape

def terminate_s_in_s(
    Smat_from: jnp.ndarray,
    z0_from: jnp.ndarray,
    Smat_into: jnp.ndarray,
    z0_into: jnp.ndarray,
    s_def = "power",
) -> jnp.ndarray:
    """
    Terminates one S-parameter matrix in another S-parameter matrix.

    Currently this only supports terminating a two-port in a one-port.
    The resultant terminated one-port S matrix is returned.

    Parameters
    ----------
    Smat_from : jnp.ndarray
        The main S matrix to terminate from.
    z0_from : jnp.ndarray
        The characteristic impedance of Smat_from.
    Smat_into : jnp.ndarray
        The main S matrix to terminate into.
    z0_into : jnp.ndarray
        The characteristic impedance of Smat_into.
    s_def : string
        The S-parameter definition.
    Returns
    -------
    jnp.ndarray
        The resulting S-parameter matrix of the terminated system.
    """
    if Smat_from.shape[1] != 2 or Smat_into.shape[1] != 1:
        raise Exception("Currently, terminate_s_in_s only allows terminating a two-port into a one-port")
    
    nfreqs = Smat_from.shape[0]
    z0_from = fix_z0_shape(z0_from, nfreqs, 2)
    z0_into = fix_z0_shape(z0_into, nfreqs, 1)

    gamma_L = Smat_into[:,0,0]
    S11 = Smat_from[:,0,0]
    S12 = Smat_from[:,0,1]
    S21 = Smat_from[:,1,0]
    S22 = Smat_from[:,1,1]

    gamma_in = S11 + (S12 * S21 * gamma_L) / (1 - S22 * gamma_L)
    S_term = gamma_in.reshape(-1, 1, 1)
    z0_term = z0_from[:,0]
    return S_term, z0_term

def terminate_a_in_s(
    Amat: jnp.ndarray,
    Smat: jnp.ndarray,
    z0: jnp.ndarray,
    s_def = "power",
) -> jnp.ndarray:
    """
    Terminates an ABCD matrix in an S-parameter matrix.

    Currently this only supports terminating a two-port in a one-port.
    The resultant terminated one-port S matrix is returned.

    Parameters
    ----------
    Amat : jnp.ndarray
        The ABCD matrix to terminate from.
    Smat : jnp.ndarray:
        The S matrix to terminate into.
    z0 : jnp.ndarray:
        The characteristic impedance of the S-matrix being terminated.
    Returns
    -------
    jnp.ndarray
        The resulting S-parameter matrix of the terminated system.
    """
    if Smat.shape[1] != 1:
        raise Exception("terminate_a_in_s can only be called for one-port S matrices")
    
    z0 = fix_z0_shape(z0, Smat.shape[0], Smat.shape[1])

    # Terminated last in s11
    s11 = Smat[:,0,0]
    
    A, B, C, D = Amat[:,0,0], Amat[:,0,1], Amat[:,1,0], Amat[:,1,1]
    num = z0 * (1 + s11) * (A - z0*C) + (B - D*z0)*(1-s11)
    den = z0 * (1 + s11) * (A + z0*C) + (B + D*z0)*(1-s11)
    s11_out = num / den        
    return s11_out.reshape(-1, 1, 1), z0[:,0]