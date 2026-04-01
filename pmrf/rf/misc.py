import jax.numpy as jnp
from typing import Any # Replace with NumberLike if imported locally

def fix_z0_shape(z0: Any, nfreqs: int, nports: int, nnetworks: int = 1) -> jnp.ndarray:
    """
    Broadcast the characteristic impedance `z0` to shape `(nfreqs, nports)` 
    if `nnetworks == 1`, or `(nnetworks, nfreqs, nports)` if `nnetworks > 1`.

    Parameters
    ----------
    z0 : NumberLike
        Input impedance. Can be a scalar, a 1D array of length `nports`,
        `nfreqs`, or `nnetworks`, a 2D array of shape `(nfreqs, nports)`,
        `(nnetworks, nports)`, or `(nnetworks, nfreqs)`, or a 3D array of
        shape `(nnetworks, nfreqs, nports)`.
    nfreqs : int
        The number of frequency points.
    nports : int
        The number of ports.
    nnetworks : int, optional
        The number of networks with identical port numbers, by default 1.

    Returns
    -------
    jnp.ndarray
        The broadcasted impedance array. Shape is `(nfreqs, nports)` 
        if `nnetworks == 1`, otherwise `(nnetworks, nfreqs, nports)`.

    Raises
    ------
    IndexError
        If `z0` has an incompatible shape.
    """
    z0_arr = jnp.asarray(z0)
    shape = z0_arr.shape
    target_shape_3d = (nnetworks, nfreqs, nports)
    
    res = None

    # Precedence explicitly ordered to match original behavior if dimensions overlap
    if shape == target_shape_3d:
        res = z0_arr
    elif shape == (nfreqs, nports):
        res = jnp.broadcast_to(z0_arr, target_shape_3d)
    elif len(shape) == 0: 
        res = jnp.full(target_shape_3d, z0_arr)
    elif len(shape) == 2:
        if shape == (nnetworks, nports):
            res = jnp.broadcast_to(z0_arr[:, None, :], target_shape_3d)
        elif shape == (nnetworks, nfreqs):
            res = jnp.broadcast_to(z0_arr[:, :, None], target_shape_3d)
    elif len(shape) == 1:
        # Original code prioritized nports over nfreqs
        if shape[0] == nports:
            res = jnp.broadcast_to(z0_arr[None, None, :], target_shape_3d)
        elif shape[0] == nfreqs:
            res = jnp.broadcast_to(z0_arr[None, :, None], target_shape_3d)
        elif shape[0] == nnetworks:
            res = jnp.broadcast_to(z0_arr[:, None, None], target_shape_3d)

    # Fallback if no matching shape was found
    if res is None:
        raise IndexError(
            f"z0 shape {shape} is not an acceptable shape for broadcasting "
            f"to (nnetworks={nnetworks}, nfreqs={nfreqs}, nports={nports})."
        )
        
    # Return 2D array for backward compatibility if nnetworks == 1
    if nnetworks == 1:
        return res[0].copy()
    
    return res.copy()