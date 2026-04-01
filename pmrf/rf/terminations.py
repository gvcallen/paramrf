import jax
import jax.numpy as jnp
from pmrf.rf.conversions import fix_z0_shape

def terminate_s_in_s(
    Smat_from: jnp.ndarray,
    z0_from: jnp.ndarray,
    Smat_into: jnp.ndarray,
    z0_into: jnp.ndarray,
    s_def = "power",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Terminates one S-parameter matrix in another S-parameter matrix.

    This supports terminating a 2N-port into an N-port, and automatically
    renormalizes the terminating network if the connecting port impedances do not match.
    The resultant terminated N-port S matrix is returned.

    Parameters
    ----------
    Smat_from : jnp.ndarray
        The main S matrix to terminate from. Shape: (nfreqs, 2N, 2N)
    z0_from : jnp.ndarray
        The characteristic impedance of Smat_from.
    Smat_into : jnp.ndarray
        The S matrix to terminate into. Shape: (nfreqs, N, N)
    z0_into : jnp.ndarray
        The characteristic impedance of Smat_into.
    s_def : string
        The S-parameter definition. Defaults to "power".
        
    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        The resulting S-parameter matrix and characteristic impedance.
    """
    N = Smat_into.shape[1]
    
    if Smat_from.shape[1] != 2 * N:
        raise ValueError(
            f"terminate_s_in_s requires a 2N-port terminating into an N-port. "
            f"Got {Smat_from.shape[1]} and {N}."
        )
    
    nfreqs = Smat_from.shape[0]
    z0_from = fix_z0_shape(z0_from, nfreqs, 2 * N)
    z0_into = fix_z0_shape(z0_into, nfreqs, N)

    # Extract block matrices
    S11 = Smat_from[:, :N, :N]
    S12 = Smat_from[:, :N, N:]
    S21 = Smat_from[:, N:, :N]
    S22 = Smat_from[:, N:, N:]
    
    # Impedances of the connecting ports
    z0_out = z0_from[:, N:]
    
    # --- Renormalization Block ---
    def apply_renorm(operand):
        S_L, z_old, z_new = operand
        
        # Power-wave reflection coefficient (diagonal per frequency)
        if s_def == "power":
            g = (z_new - z_old) / (z_new + jnp.conj(z_old))
        else:
            g = (z_new - z_old) / (z_new + z_old)
            
        # Convert to diagonal matrices: (nfreqs, N, N)
        G = jax.vmap(jnp.diag)(g)
        I = jnp.eye(N, dtype=S_L.dtype)
        
        I_minus_G = I - G
        
        # Calculate S_new = (I - G)^-1 @ (S_L - G) @ (I - G @ S_L)^-1 @ (I - G)
        # We use jnp.linalg.solve to avoid explicit inverses
        X = jnp.linalg.solve(I_minus_G, S_L - G)          # X = (I - G)^-1 @ (S_L - G)
        Z = jnp.linalg.solve(I - G @ S_L, I_minus_G)      # Z = (I - G @ S_L)^-1 @ (I - G)
        
        return X @ Z

    def skip_renorm(operand):
        S_L, _, _ = operand
        return S_L

    # Check if we need to renormalize (using allclose for floating point safety)
    needs_renorm = jnp.logical_not(jnp.allclose(z0_out, z0_into))
    
    # jax.lax.cond strictly preserves the fast path during JIT execution
    S_L_matched = jax.lax.cond(
        needs_renorm,
        apply_renorm,
        skip_renorm,
        (Smat_into, z0_into, z0_out)
    )
    # -----------------------------

    I = jnp.eye(N, dtype=Smat_from.dtype)

    # Compute S_term = S11 + S12 @ S_L @ inv(I - S22 @ S_L) @ S21
    diff = I - S22 @ S_L_matched
    X = jnp.linalg.solve(diff, S21)
    
    S_term = S11 + S12 @ S_L_matched @ X
    z0_term = z0_from[:, :N]
    
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