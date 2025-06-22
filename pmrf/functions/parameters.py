import pmrf.numpy as np
from pmrf.numpy import USE_JAX
from pmrf._misc import NumberLike

if USE_JAX:
    import jax
    from jax import lax
    from jax.scipy.special import gammaln
    from jax._src.numpy.ufuncs import _constant_like
else:
    from scipy.special import gammaln
    
def a2s(a: np.ndarray, z0: NumberLike = 50) -> np.ndarray:
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    nfreqs, nports, nports = a.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = _fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    A = a[:,0,0]
    B = a[:,0,1]
    C = a[:,1,0]
    D = a[:,1,1]
    denom = A*z02 + B + C*z01*z02 + D*z01

    s = np.array([
        [
            (A*z02 + B - C*z01.conj()*z02 - D*z01.conj() ) / denom,
            (2*np.sqrt(z01.real * z02.real)) / denom,
        ],
        [
            (2*(A*D - B*C)*np.sqrt(z01.real * z02.real)) / denom,
            (-A*z02.conj() + B - C*z01*z02.conj() + D*z01) / denom,
        ],
    ]).transpose()
    return s

def s2a(s: np.ndarray, z0: NumberLike = 50) -> np.ndarray:
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    nfreqs, nports, nports = s.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = _fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    denom = (2*s[:,1,0]*np.sqrt(z01.real * z02.real))
    a = np.array([
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

def _fix_z0_shape(z0: NumberLike, nfreqs: int, nports: int) -> np.ndarray:
    # Taken from scikit-rf. See the copyright notice in pmrf._frequency.py
    if np.shape(z0) == (nfreqs, nports):
        return z0.copy()
    elif np.ndim(z0) == 0:
        return np.array(nfreqs * [nports * [z0]])
    elif len(z0) == nports:
        return np.array(nfreqs * [z0])
    elif len(z0) == nfreqs:
        return np.array(nports * [z0]).T
    else:
        raise IndexError('z0 is not an acceptable shape')