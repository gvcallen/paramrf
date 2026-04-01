"""
Core math functions.
"""
import types

from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy import imag, pi, real, unwrap
from jax import lax
from jax.scipy.special import gammaln
from jax._src.numpy.ufuncs import _constant_like

from pmrf.constants import NumberLike, INF, LOG_OF_NEG

def rsolve(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    r"""
    Solves x @ A = B.

    Calls `numpy.linalg.solve` with transposed matrices.
    Equivalent to `B @ np.linalg.inv(A)` but avoids calculating the inverse explicitely.

    Input should have dimension of similar to (nfreqs, nports, nports).

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
    return jnp.transpose(jnp.linalg.solve(jnp.transpose(A, (0, 2, 1)).conj(),
            jnp.transpose(B, (0, 2, 1)).conj()), (0, 2, 1)).conj()

def nudge_eig(
    mat: jnp.ndarray,
    cond: float = 1e-9,
    min_eig: float = 1e-12,
) -> jnp.ndarray:
    r"""
    Nudge eigenvalues with absolute value smaller than `max(cond * max(eigenvalue), min_eig)` to that value.
    
    Can be used to avoid singularities in solving matrix equations.
    Input should have dimension of similar to (nfreqs, nports, nports).

    Parameters
    ----------
    mat : np.ndarray
        Matrices to nudge.
    cond : float, optional
        Minimum eigenvalue ratio compared to the maximum eigenvalue.
        Default value is `1e-9`.
    min_eig : float, optional
        Minimum eigenvalue.
        Default value is `1e-12`.

    Returns
    -------
    res : np.ndarray
        Nudged matrices.
    """
    eigw, eigv = jnp.linalg.eig(mat)
    max_eig = jnp.amax(jnp.abs(eigw), axis=1)
    mask = jnp.logical_or(jnp.abs(eigw) < cond * max_eig[:, None], jnp.abs(eigw) < min_eig)
    has_problem = mask.any()
    
    def fix_branch():
        nonlocal eigw
        # mask_cond = cond * jnp.repeat(max_eig[:, None], mat.shape[-1], axis=-1)[mask]
        # mask_min = min_eig * jnp.ones(mask_cond.shape)
        # eigw[mask] = jnp.maximum(mask_cond, mask_min)
        mask_array = cond * jnp.repeat(max_eig[:, None], mat.shape[-1], axis=-1)
        mask_min = min_eig * jnp.ones_like(mask_array)
        eigw = jnp.where(mask, jnp.maximum(mask_array, mask_min), eigw)

        # Now assemble the eigendecomposited matrices back
        e = jnp.zeros_like(mat)
        # other = jnp.einsum('ijj->ij', e)
        # e = jnp.einsum('ijj->ij', e).at[...].set(eigw)
        # e = e.at[jnp.diag_indices(e.shape[1], e.shape[2])].set(eigw)
        rows = jnp.arange(e.shape[1])
        e = e.at[jnp.arange(e.shape[0])[:, None], rows, rows].set(eigw)

        return rsolve(eigv, eigv @ e)
    
    def no_fix_branch():
        return mat
    
    return jax.lax.cond(has_problem, fix_branch, no_fix_branch)

def nudge_svd(mat: jnp.ndarray, 
              cond: float = 1e-9, 
              min_val: float = 1e-12) -> jnp.ndarray:
    """
    Nudge small singular values to avoid singularities using SVD.
    SVD is natively supported by JAX autodiff for non-symmetric matrices.
    """
    # 1. Decompose: mat = U * S * Vh
    U, S, Vh = jnp.linalg.svd(mat)
    
    # 2. Find the threshold (S is always strictly real and non-negative)
    max_s = jnp.amax(S, axis=-1, keepdims=True)
    threshold = jnp.maximum(cond * max_s, min_val)
    
    # 3. Nudge singular values that fall below the threshold
    S_nudged = jnp.maximum(S, threshold)
    
    # 4. Reconstruct and return the conditioned matrix
    # S_nudged[..., None] broadcasts the 1D array to multiply rows of Vh correctly
    return U @ (S_nudged[..., None] * Vh)

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

def round_sig(x, sig=3):
    """
    Round to a specific number of significant digits.

    Parameters
    ----------
    x : float
        Number to round.
    sig : int, optional, default=3
        Number of significant digits.

    Returns
    -------
    float
        Rounded number.
    """
    if x == 0:
        return 0
    return round(x, sig - int(jnp.floor(jnp.log10(abs(x)))) - 1)

def comb(N: jnp.ndarray, k: jnp.ndarray, exact: bool = False, repetition: bool = False):
    r"""
    The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    When exact=False, the result is approximately and efficiently computed using the following formula:
    
    .. math::
    
        \exp\left\{\ln{\Gamma(N+1)} - [\ln{\Gamma(k+1)} + \ln{\Gamma(N+1-k)}]\right\}

    using the Gamma function. 


    Parameters
    ----------
    N : np.ndarray
        The number of things.
    k : np.ndarray
        The number of elements taken.
    exact : bool, optional
        If `True`, the result is computed exactly and returned as an integer type.
        Currently, vectorization is not supported for exact=True.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    comb : np.ndarray
        The number of combinations of N things taken k at a time.
    """
    if repetition:
        return comb(N + k - 1, k, exact=exact, repetition=False)
  
    if exact:
        max_divisor = lax.max(k, N - k)
        min_divisor = lax.min(k, N - k)
        N_factorial_over_max_factorial = jnp.prod(jnp.arange(N, max_divisor, -1))
        return lax.div(N_factorial_over_max_factorial, jnp.prod(jnp.arange(1, min_divisor + 1)))

    one = _constant_like(N, 1)
    N_plus_1 = lax.add(N,one)
    k_plus_1 = lax.add(k,one)
    return lax.exp(lax.sub(gammaln(N_plus_1),lax.add(gammaln(k_plus_1), gammaln(lax.sub(N_plus_1,k)))))

def evaluate_power_basis(x, coeffs, lower_bound, upper_bound):
    """
    Evaluate a polynomial in the power basis.

    The input `x` is normalized to the range `[0, 1]` based on bounds.

    Parameters
    ----------
    x : jnp.ndarray
        Input values.
    coeffs : jnp.ndarray
        Polynomial coefficients.
    lower_bound : float
        Lower bound of the domain.
    upper_bound : float
        Upper bound of the domain.

    Returns
    -------
    jnp.ndarray
        Evaluated polynomial values.
    """
    coeffs = jnp.asarray(coeffs)
    x_norm = (x - lower_bound) / (upper_bound - lower_bound)
    return jnp.polyval(coeffs[::-1], x_norm)

def evaluate_bernstein_basis(x, coeffs, lower_bound, upper_bound):
    """
    Evaluate a polynomial in the Bernstein basis.

    Parameters
    ----------
    x : jnp.ndarray
        Input values.
    coeffs : jnp.ndarray
        Coefficients (control points) of the Bernstein polynomial.
    lower_bound : float
        Lower bound of the domain.
    upper_bound : float
        Upper bound of the domain.

    Returns
    -------
    jnp.ndarray
        Evaluated values.
    """
    coeffs = jnp.asarray(coeffs)
    n = len(coeffs) - 1  # Degree of the polynomial

    i = jnp.arange(n + 1)
    binomial_coeffs = comb(n, i)

    t = (x - lower_bound) / (upper_bound - lower_bound)

    def _eval_single(t_scalar):
        basis_values = jnp.power(t_scalar, i) * jnp.power(1 - t_scalar, n - i)
        return jnp.dot(coeffs, binomial_coeffs * basis_values)

    
    result = jax.vmap(_eval_single)(jnp.atleast_1d(t))
    return result

def broaden(key, x, percentage=0.1):
    """
    Broaden data using gaussian noise by a specified percentage.
    
    The broadening is relative to the standard deviation of each column in the data.
    Since standard deviations add with the square, random noise is added
    with noise_std = data_std * (sqrt ((1.0 + percentage)**2) - 1.0).

    Parameters
    ----------
    x : jnp.ndarray
        Input data.
    percentage : float
        The percentage to broaden the data by.

    Returns
    -------
    jnp.ndarray
        The broadened data.
    """    
    ratio = 1 + percentage
    scale = jnp.sqrt(ratio**2 - 1)
    
    stds = jnp.std(x, axis=0, keepdims=True)
    noise = jax.random.normal(key, shape=x.shape)
    scaled_noise = noise * stds * scale
    return x + scaled_noise


def unwrap_rad(phi: NumberLike):
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
    return unwrap(phi, axis=0)


def sqrt_known_sign(z_squared: NumberLike, z_approx: NumberLike):
    """
    Return the square root of a complex number, with sign chosen to match `z_approx`.

    Parameters
    ----------
    z_squared : number or array-like
        The complex to be square-rooted.
    z_approx : number or array-like
        The approximate value of z. The sign of z is chosen to match that of
        z_approx.

    Returns
    -------
    z : number, array-like (same type as z_squared)
        Square root of z_squared.
    """
    z = jnp.sqrt(z_squared)
    return jnp.where(
        jnp.sign(jnp.angle(z)) == jnp.sign(jnp.angle(z_approx)),
        z, z.conj())


def find_correct_sign(z1: NumberLike, z2: NumberLike, z_approx: NumberLike):
    r"""
    Create new vector from z1, z2 choosing elements with sign matching z_approx.

    This is used when you have to make a root choice on a complex number.
    and you know the approximate value of the root.

    .. math::

        z1,z2 = \pm \sqrt(z^2)


    Parameters
    ----------
    z1 : array-like
        Root 1.
    z2 : array-like
        Root 2.
    z_approx : array-like
        Approximate answer of z.

    Returns
    -------
    z3 : np.array
        Array built from z1 and z2 by
        z1 where sign(z1) == sign(z_approx), z2 else.
    """
    return jnp.where(
    jnp.sign(jnp.angle(z1)) == jnp.sign(jnp.angle(z_approx)),z1, z2)


def find_closest(z1: NumberLike, z2: NumberLike, z_approx: NumberLike):
    """
    Return z1 or z2 depending on which is closer to z_approx.

    Parameters
    ----------
    z1 : array-like
        Root 1.
    z2 : array-like
        Root 2.
    z_approx : array-like
        Approximate answer of z.

    Returns
    -------
    z3 : np.array
        Array built from z1 and z2.
    """
    z1_dist = abs(z1-z_approx)
    z2_dist = abs(z2-z_approx)

    return jnp.where(z1_dist<z2_dist,z1, z2)

def sqrt_phase_unwrap(z: NumberLike):
    r"""
    Take the square root of a complex number with unwrapped phase.

    This idea came from Lihan Chen.

    .. math::

        \sqrt{|z|} \exp( \arg_{unwrap}(z) / 2 )


    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    z : number of array_like
        A complex number or sequence of complex numbers.
    """
    return jnp.sqrt(abs(z))*\
            jnp.exp(0.5*1j*unwrap_rad(complex_2_radian(z)))


# mathematical functions
def dirac_delta(x: NumberLike):
    r"""
    Calculate Dirac function.

    Dirac function :math:`\delta(x)` defined as :math:`\delta(x)=1` if x=0,
    0 otherwise.

    Parameters
    ----------
    x : number of array_like
        A real number or sequence of real numbers.

    Returns
    -------
    delta : number of array_like
        1 or 0.

    References
    ----------
    https://en.wikipedia.org/wiki/Dirac_delta_function
    """
    return (x==0)*1. + (x!=0)*0.


def neuman(x: NumberLike):
    r"""
    Calculate Neumans number.

    It is defined as:

    .. math::

        2 - \delta(x)

    where :math:`\delta` is the Dirac function.

    Parameters
    ----------
    x : number or array_like
        A real number or sequence of real numbers.

    Returns
    -------
    y : number or array_like
        A real number or sequence of real numbers.

    See Also
    --------
    dirac_delta
    """
    return 2. - dirac_delta(x)


def null(A: jnp.ndarray, eps: float = 1e-15):
    """
    Calculate the null space of matrix A.

    Parameters
    ----------
    A : array_like
        Input matrix.
    eps : float, optional, default=1e-15
        Epsilon value for singular value thresholding.

    Returns
    -------
    null_space : array_like
        The null space of A.

    References
    ----------
    https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
    """
    u, s, vh = jnp.linalg.svd(A)
    null_space = jnp.compress(s <= eps, vh, axis=0)
    return null_space.T


def inf_to_num(x: NumberLike):
    """
    Convert inf and -inf's to large numbers.

    Parameters
    ----------
    x : array-like or number
        The input array or number.

    Returns
    -------
    x : Number of array_like
        Input without with +/- inf replaced by large numbers.
    """
    x = jnp.nan_to_num(x, nan=jnp.nan, posinf=INF, neginf=-1*INF)
    return x


def cross_ratio(a: NumberLike, b: NumberLike, c: NumberLike, d:NumberLike):
    r"""
    Calculate the cross ratio of a quadruple of distinct points on the real line.

    The cross ratio is defined as:

    .. math::

        r = \frac{ (a-b)(c-d) }{ (a-d)(c-b) }

    Parameters
    ----------
    a,b,c,d : array-like or number
        Input points.

    Returns
    -------
    r : array-like or number
        The cross ratio.

    References
    ----------
    https://en.wikipedia.org/wiki/Cross-ratio
    """
    return ((a-b)*(c-d))/((a-d)*(c-b))


def complexify(f: Callable, name: str = None):
    """
    Make a function f(scalar) into f(complex).

    If `f(x)` then it returns `f_c(z) = f(real(z)) + 1j*f(imag(z))`

    If the real/imag arguments are not first, then you may specify the
    name given to them as kwargs.

    Parameters
    ----------
    f : Callable
        Function of real variable.
    name : string, optional
        Name of the real/imag argument names if they are not first.

    Returns
    -------
    f_c : Callable
        Function of a complex variable.

    Examples
    ----------
    >>> def f(x): return x
    >>> f_c = rf.complexify(f)
    >>> z = 0.2 -1j*0.3
    >>> f_c(z)
    """
    def f_c(z, *args, **kw):
        if name is not None:
            kw_re = {name: real(z)}
            kw_im = {name: imag(z)}
            kw_re.update(kw)
            kw_im.update(kw)
            return f(*args, **kw_re) + 1j*f(*args, **kw_im)
        else:
            return f(real(z), *args,**kw) + 1j*f(imag(z), *args, **kw)
    return f_c


def multiply_by(x: jnp.ndarray, by: jnp.ndarray, axis=None) -> jnp.ndarray:
    """
    Broadcast multiply array `x` by array `by` along a specified axis.
    
    This function tiles `by` to match the dimensions of `x` implied by block multiplication.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    by : jnp.ndarray
        Array to multiply with.
    axis : int, optional
        Axis along which to apply the multiplication (0 for rows, 1 for columns).
    
    Returns
    -------
    jnp.ndarray
        Result of the multiplication.
        
    Raises
    ------
    ValueError
        If the shape of the specified axis is not divisible by the length of `by`.
    """
    if by.shape == x.shape:
        x *= by
    else:
        n = len(by)

        if len(x.shape) == 1:
            if axis == 0:
                x = x.reshape(len(x), 1)
            else:
                x = x.reshape(1, len(x))
        
        if axis and x.shape[axis] % n != 0:
            raise ValueError(f"The length of the specified axis ({x.shape[axis]}) is not divisible by {n}.")

        if axis == 0:
            by = jnp.tile(by, (int(x.shape[0] / n), x.shape[1]))
        elif axis == 1:
            by = jnp.tile(by, (x.shape[0], int(x.shape[1] / n)))

        x *= by

    return x

def sum_every(x: jnp.ndarray, n: int, axis=None) -> jnp.ndarray:
    """
    Sum blocks of size `n` along a specified axis.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    n : int
        Block size to sum.
    axis : int, optional
        Axis along which to sum (0 for rows, 1 for columns).

    Returns
    -------
    jnp.ndarray
        Resulting summed array.

    Raises
    ------
    ValueError
        If the shape of the specified axis is not divisible by `n`.
    """
    if len(x.shape) == 1 and len(x) % n == 0:
        if axis == 0:
            x = x.reshape(len(x), 1)
        else:
            x = x.reshape(1, len(x))

    # Ensure the axis length is divisible by i
    if x.shape[axis] % n != 0:
        raise ValueError(f"The length of the specified axis ({x.shape[axis]}) is not divisible by {n}.")

    # Reshape and sum
    if axis == 0:
        # Group rows
        x = x.reshape(-1, n, x.shape[1]).sum(axis=1)
    elif axis == 1:
        # Group columns
        x = x.reshape(x.shape[0], -1, n).sum(axis=2)

    return x

def multiply_every(x: jnp.ndarray, n: int, axis=None) -> jnp.ndarray:
    """
    Multiply blocks of size `n` along a specified axis.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    n : int
        Block size to multiply.
    axis : int, optional
        Axis along which to multiply (0 for rows, 1 for columns).

    Returns
    -------
    jnp.ndarray
        Resulting product array.

    Raises
    ------
    ValueError
        If the shape of the specified axis is not divisible by `n`.
    """
    if len(x.shape) == 1 and len(x) % n == 0:
        if axis == 0:
            x = x.reshape(len(x), 1)
        else:
            x = x.reshape(1, len(x))

    # Ensure the axis length is divisible by i
    if x.shape[axis] % n != 0:
        raise ValueError(f"The length of the specified axis ({x.shape[axis]}) is not divisible by {n}.")

    # Reshape and sum
    if axis == 0:
        # Group rows
        x = x.reshape(-1, n, x.shape[1]).prod(axis=1)
    elif axis == 1:
        # Group columns
        x = x.reshape(x.shape[0], -1, n).prod(axis=2)

    return x