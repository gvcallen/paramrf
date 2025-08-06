import jax.numpy as jnp
from pmrf._constants import NumberLike

import jax
from jax import lax
from jax.scipy.special import gammaln
from jax._src.numpy.ufuncs import _constant_like
    
def a2s(a: jnp.ndarray, z0: NumberLike = 50) -> jnp.ndarray:
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

def renormalize_s(
    s: jnp.ndarray,
    z_old: NumberLike,
    z_new: NumberLike,
) -> jnp.ndarray:
    """
    Renormalize S-parameters from z_old to z_new impedances.

    Parameters
    ----------
    s : jnp.ndarray, shape (nfreqs, nports, nports)
        S-parameter matrix.
    z_old : scalar, (nports,), or (nfreqs, nports)
    z_new : scalar, (nports,), or (nfreqs, nports)

    Returns
    -------
    jnp.ndarray of shape (nfreqs, nports, nports)
        Renormalized S-parameters.
    """
    nfreqs, nports, _ = s.shape

    Z_A = fix_z0_shape(z_old, nfreqs, nports)  # shape: (nfreqs, nports)
    Z_B = fix_z0_shape(z_new, nfreqs, nports)

    # Check if z_old and z_new are frequency-independent
    freq_indep = (
        jnp.ndim(z_old) == 0 or jnp.shape(z_old) == (nports,)
    ) and (
        jnp.ndim(z_new) == 0 or jnp.shape(z_new) == (nports,)
    )

    if freq_indep:
        z_a = Z_A[0]  # shape: (nports,)
        z_b = Z_B[0]

        z_prod_sqrt_inv = 1.0 / jnp.sqrt(z_a * z_b)
        D = 0.5 * jnp.diag(z_prod_sqrt_inv)

        M_s = D @ jnp.linalg.inv(jnp.diag(z_a + z_b))
        M_c = D @ jnp.linalg.inv(jnp.diag(z_a - z_b))

        def renorm_fixed(s_f):
            return (M_c + M_s @ s_f) @ jnp.linalg.inv(M_s + M_c @ s_f)

        return jax.vmap(renorm_fixed)(s)

    else:
        def renorm_per_freq(s_f, z_a, z_b):
            z_prod_sqrt_inv = 1.0 / jnp.sqrt(z_a * z_b)
            D = 0.5 * jnp.diag(z_prod_sqrt_inv)

            M_s = D @ jnp.linalg.inv(jnp.diag(z_a + z_b))
            M_c = D @ jnp.linalg.inv(jnp.diag(z_a - z_b))

            return (M_c + M_s @ s_f) @ jnp.linalg.inv(M_s + M_c @ s_f)

        return jax.vmap(renorm_per_freq, in_axes=(0, 0, 0))(s, Z_A, Z_B)

def fix_z0_shape(z0: NumberLike, nfreqs: int, nports: int) -> jnp.ndarray:
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    if jnp.shape(z0) == (nfreqs, nports):
        return z0.copy()
    elif jnp.ndim(z0) == 0:
        return jnp.array(nfreqs * [nports * [z0]])
    elif len(z0) == nports:
        return jnp.array(nfreqs * [z0])
    elif len(z0) == nfreqs:
        return jnp.array(nports * [z0]).T
    else:
        raise IndexError('z0 is not an acceptable shape')
        