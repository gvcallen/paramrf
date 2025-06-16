from pmrf._numpy import numpy as np
from pmrf._numpy import USE_JAX

if USE_JAX:
    import jax
    from jax import lax
    from jax.scipy.special import gammaln
    from jax._src.numpy.ufuncs import _constant_like
else:
    from scipy.special import gammaln


def dB20(values):
    return 20 * np.log10(np.abs(values))

def dB20inv(values):
    return 10 ** (values / 20)

def p2r(radii, angles, deg=False):
    if deg:
        angles = np.deg2rad(angles)
    return radii * np.exp(1j*angles)

def r2p(x, deg=False):
    return abs(x), np.angle(x, deg=deg)

def rms(values):
    return np.sqrt(np.mean(values**2))

def norm(y: np.ndarray, mode='L2', axis=None):
    if mode == 'Linf':
        y = np.linalg.norm(y, ord=np.inf, axis=axis)        # max(abs(y))
    elif mode == 'L1':
        y = np.linalg.norm(y, ord=1, axis=axis)             # sum(abs(y))
    elif mode == 'L2':
        y = np.linalg.norm(y, ord=2, axis=axis)             # sqrt(sum(abs(y)**2))
    elif mode == 'L2sqr':
        y = np.linalg.norm(y, ord=2, axis=axis)**2          # sum(abs(y)**2)
    else:
        raise Exception('Unknown norm type')
    
    return y

def round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def comb(N: np.ndarray, k: np.ndarray, exact: bool = False, repetition: bool = False):
    r"""The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Args:
    N: The number of things.
    k: The number of elements taken.
    exact: If `True`, the result is computed exactly and returned as an integer type.
        Currently, vectorization is not supported for exact=True.
    repetition: If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns:
    The number of combinations of N things taken k at a time.

    Notes:
    When exact=False, the result is approximately and efficiently computed using the following formula:

    .. math::
    \begin{equation}
    \exp\left\{\ln{\Gamma(N+1)} - [\ln{\Gamma(k+1)} + \ln{\Gamma(N+1-k)}]\right\}
    \end{equation}

    Where we use the Gamma function. 
    """
    if repetition:
        return comb(N + k - 1, k, exact=exact, repetition=False)
  
    if exact:
        max_divisor = lax.max(k, N - k)
        min_divisor = lax.min(k, N - k)
        N_factorial_over_max_factorial = np.prod(np.arange(N, max_divisor, -1))
        return lax.div(N_factorial_over_max_factorial, np.prod(np.arange(1, min_divisor + 1)))

    one = _constant_like(N, 1)
    N_plus_1 = lax.add(N,one)
    k_plus_1 = lax.add(k,one)
    return lax.exp(lax.sub(gammaln(N_plus_1),lax.add(gammaln(k_plus_1), gammaln(lax.sub(N_plus_1,k)))))

def evaluate_power_basis(x, coeffs, lower_bound, upper_bound):
    coeffs = np.asarray(coeffs)
    x_norm = (x - lower_bound) / (upper_bound - lower_bound)
    return np.polyval(coeffs[::-1], x_norm)

def evaluate_bernstein_basis(x, coeffs, lower_bound, upper_bound):
    coeffs = np.asarray(coeffs)
    n = len(coeffs) - 1  # Degree of the polynomial

    i = np.arange(n + 1)
    binomial_coeffs = comb(n, i)

    t = (x - lower_bound) / (upper_bound - lower_bound)

    def _eval_single(t_scalar):
        basis_values = np.power(t_scalar, i) * np.power(1 - t_scalar, n - i)
        return np.dot(coeffs, binomial_coeffs * basis_values)

    if USE_JAX:
        return jax.vmap(_eval_single)(np.atleast_1d(t))