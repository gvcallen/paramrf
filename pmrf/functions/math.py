from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy import imag, pi, real, unwrap
from jax import lax
from jax.scipy.special import gammaln
from jax._src.numpy.ufuncs import _constant_like

from pmrf.constants import NumberLike, INF, LOG_OF_NEG

def complex_2_magnitude(z: NumberLike):
    """
    Return the magnitude of the complex argument.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    mag : ndarray or scalar
        The absolute value of the input.
    """
    return jnp.abs(z)


def complex_2_db(z: NumberLike):
    r"""
    Return the magnitude in dB of a complex number (as :math:`20\log_{10}(|z|)`).

    The magnitude in dB is defined as :math:`20\log_{10}(|z|)`
    where :math:`z` is a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    mag20dB : ndarray or scalar
        The magnitude in decibels.
    """
    return magnitude_2_db(jnp.abs(z))


def complex_2_db10(z: NumberLike):
    r"""
    Return the magnitude in dB of a complex number (as :math:`10\log_{10}(|z|)`).

    The magnitude in dB is defined as :math:`10\log_{10}(|z|)`
    where :math:`z` is a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    mag10dB : ndarray or scalar
        The magnitude in decibels (power factor 10).
    """
    return mag_2_db10(jnp.abs(z))


def complex_2_radian(z: NumberLike):
    """
    Return the angle complex argument in radian.
    
    

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    ang_rad : ndarray or scalar
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.
    """
    return jnp.angle(z)


def complex_2_degree(z: NumberLike):
    """
    Returns the angle complex argument in degrees.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    ang_deg : ndarray or scalar
        The angle in degrees.
    """
    return jnp.angle(z, deg=True)


def complex_2_quadrature(z: NumberLike):
    r"""
    Take a complex number and returns quadrature, which is (length, arc-length from real axis).

    Arc-length is calculated as :math:`|z| \arg(z)`.
    
    

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    mag : array like or scalar
        Magnitude (length).
    arc_length : array like or scalar
        Arc-length from real axis: angle * magnitude.
    """
    return (jnp.abs(z), jnp.angle(z)*jnp.abs(z))


def complex_2_reim(z: NumberLike):
    """
    Return real and imaginary parts of a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    real : array like or scalar
        Real part of input.
    imag : array like or scalar
        Imaginary part of input.
    """
    return (jnp.real(z), jnp.imag(z))


def complex_components(z: NumberLike):
    """
    Break up a complex array into all possible scalar components.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    c_real : array like or scalar
        Real part.
    c_imag : array like or scalar
        Imaginary part.
    c_angle : array like or scalar
        Angle in degrees.
    c_mag : array like or scalar
        Magnitude.
    c_arc : array like or scalar
        Arclength from real axis, angle*magnitude.
    """
    return (*complex_2_reim(z), jnp.angle(z,deg=True), *complex_2_quadrature(z))


def magnitude_2_db(z: NumberLike, zero_nan: bool = True):
    """
    Convert linear magnitude to dB.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.
    zero_nan : bool, optional, default=True
        Replace NaN with zero.

    Returns
    -------
    z : number or array_like
        Magnitude in dB given by :math:`20 \log_{10}(|z|)`.
    """
    out = 20 * jnp.log10(z)
    if zero_nan:
        return jnp.nan_to_num(out, nan=LOG_OF_NEG, neginf=-jnp.inf)
    return out

mag_2_db = magnitude_2_db


def mag_2_db10(z: NumberLike, zero_nan:bool = True):
    """
    Convert linear magnitude to dB (factor 10).

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    zero_nan : bool, optional, default=True
        Replace NaN with zero.

    Returns
    -------
    z : array_like
        Magnitude in dB given by :math:`10 \log_{10}(|z|)`.
    """
    out = 10 * jnp.log10(z)
    if zero_nan:
        return jnp.nan_to_num(out, nan=LOG_OF_NEG, neginf=-jnp.inf)
    return out


def db_2_magnitude(z: NumberLike):
    """
    Convert dB to linear magnitude.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    z : number or array_like
        10**((z)/20) where z is a complex number.
    """
    return 10**((z)/20.)

db_2_mag = db_2_magnitude


def db10_2_mag(z: NumberLike):
    """
    Convert dB (factor 10) to linear magnitude.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    z : array_like
        10**((z)/10) where z is a complex number.
    """
    return 10**((z)/10.)


def magdeg_2_reim(mag: NumberLike, deg: NumberLike):
    """
    Convert linear magnitude and phase (in deg) arrays into a complex array.

    Parameters
    ----------
    mag : number or array_like
        Magnitude.
    deg : number or array_like
        Phase in degrees.

    Returns
    -------
    z : array_like
        A complex number or sequence of complex numbers.
    """
    return mag*jnp.exp(1j*deg*pi/180.)

def dbdeg_2_reim(db: NumberLike, deg: NumberLike):
    """
    Convert dB magnitude and phase (in deg) arrays into a complex array.

    Parameters
    ----------
    db : number or array_like
        Magnitude in dB.
    deg : number or array_like
        Phase in degrees.

    Returns
    -------
    z : array_like
        A complex number or sequence of complex numbers.
    """
    return magdeg_2_reim(db_2_magnitude(db), deg)


def db_2_np(db: NumberLike):
    """
    Convert a value in decibel (dB) to neper (Np).

    Parameters
    ----------
    db : number or array_like
        A real number or sequence of real numbers.

    Returns
    -------
    np : number or array_like
        A real number of sequence of real numbers.
    """
    return (jnp.log(10)/20) * db


def np_2_db(x: NumberLike):
    """
    Convert a value in Nepers (Np) to decibel (dB).

    Parameters
    ----------
    x : number or array_like
        A real number or sequence of real numbers.

    Returns
    -------
    db : number or array_like
        A real number of sequence of real numbers.
    """
    return 20/jnp.log(10) * x


def radian_2_degree(rad: NumberLike):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    rad : number or array_like
        Angle in radian.

    Returns
    -------
    deg : number or array_like
        Angle in degree.
    """
    return (rad)*180/pi


def degree_2_radian(deg: NumberLike):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    deg : number or array_like
        Angle in degrees.

    Returns
    -------
    rad : number or array_like
        Angle in radians.
    """
    return (deg)*pi/180.


def feet_2_meter(feet: NumberLike = 1):
    """
    Convert length in feet to meter.

    1 foot is equal to 0.3048 meters.

    Parameters
    ----------
    feet : number or array-like, optional, default=1
        Length in feet.

    Returns
    -------
    meter: number or array-like
        Length in meter.

    See Also
    --------
    meter_2_feet
    """
    return 0.3048*feet

def meter_2_feet(meter: NumberLike = 1):
    """
    Convert length in meter to feet.

    1 meter is equal to 3.28084 feet.

    Parameters
    ----------
    meter : number or array-like, optional, default=1
        Length in meter.

    Returns
    -------
    feet : number or array-like
        Length in feet.

    See Also
    --------
    feet_2_meter
    """
    return 3.28084*meter


def db_per_100feet_2_db_per_100meter(db_per_100feet: NumberLike = 1):
    """
    Convert attenuation values given in dB/100ft to dB/100m.

    db_per_100meter = db_per_100feet * rf.meter_2_feet()

    Parameters
    ----------
    db_per_100feet : number or array-like, optional, default=1
        Attenuation in dB/100 ft.

    Returns
    -------
    db_per_100meter : number or array-like
        Attenuation in dB/100 m.

    See Also
    --------
    meter_2_feet
    feet_2_meter
    np_2_db
    db_2_np
    """
    return db_per_100feet * 100 / feet_2_meter(100)


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



# old functions just for reference
def complex2Scalar(z: NumberLike):
    """
    Serialize a list/array of complex numbers.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    re_im : array_like
        Produce the following array for an input z:
        z[0].real, z[0].imag, z[1].real, z[1].imag, etc.

    See Also
    --------
    scalar2Complex
    """
    z = jnp.array(z)
    re_im = []
    for k in z:
        re_im.append(jnp.real(k))
        re_im.append(jnp.imag(k))
    return jnp.array(re_im).flatten()

def scalar2Complex(s: NumberLike):
    """
    Unserialize a list/array of real and imag numbers into a complex array.

    Inverse of :func:`complex2Scalar`.

    Parameters
    ----------
    s : array_like
        An array with real and imaginary parts ordered as:
        z[0].real, z[0].imag, z[1].real, z[1].imag, etc.

    Returns
    -------
    z : Number or array_like
        A complex number or sequence of complex number.

    See Also
    --------
    complex2Scalar
    """
    s = jnp.array(s)
    z = []

    for k in range(0,len(s),2):
        z.append(s[k] + 1j*s[k+1])
    return jnp.array(z).flatten()    

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

def convolve_interleaved(x, axis=1) -> jnp.ndarray:
    """
    Convolve interleaved signals (e.g. real/imag parts stored sequentially).

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    axis : int, optional, default=1
        Axis along which to convolve.

    Returns
    -------
    jnp.ndarray
        The convolved result.
    
    Raises
    ------
    Exception
        If the input is not 1D or axis is not 1 (not yet implemented).
    """
    if axis == 1:
        if len(x.shape) == 1:
            y1, y2 = x[0::2], x[1::2]
            x = jnp.convolve(y1, y2)
        else:
            raise Exception("Not yet implemented")
    else:
        raise Exception("Not yet implemented")

    return x

def rms(x):
    """
    Calculate the Root Mean Square (RMS) of the input.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    float or jnp.ndarray
        The RMS value.
    """
    return jnp.sqrt(jnp.mean(x**2))

def mag_2_db(x):
    """
    Convert magnitude to decibels.

    Parameters
    ----------
    x : jnp.ndarray
        Linear magnitude.

    Returns
    -------
    jnp.ndarray
        Value in dB.
    """
    return 20 * jnp.log10(jnp.abs(x))

def db_2_mag(x):
    """
    Convert decibels to magnitude.

    Parameters
    ----------
    x : jnp.ndarray
        Value in dB.

    Returns
    -------
    jnp.ndarray
        Linear magnitude.
    """
    return 10 ** (x / 20)

def polar_2_rect(radii, angles, deg=False):
    """
    Convert polar coordinates to rectangular (complex) coordinates.

    Parameters
    ----------
    radii : jnp.ndarray
        Radius (magnitude).
    angles : jnp.ndarray
        Angle (phase).
    deg : bool, optional, default=False
        If True, angles are in degrees.

    Returns
    -------
    jnp.ndarray
        Complex number in rectangular coordinates.
    """
    if deg:
        angles = jnp.deg2rad(angles)
    return radii * jnp.exp(1j*angles)

def rect_2_polar(x, deg=False):
    """
    Convert rectangular (complex) coordinates to polar coordinates.

    Parameters
    ----------
    x : jnp.ndarray
        Complex number input.
    deg : bool, optional, default=False
        If True, return angle in degrees.

    Returns
    -------
    radii : jnp.ndarray
        Magnitude.
    angles : jnp.ndarray
        Phase angle.
    """
    return abs(x), jnp.angle(x, deg=deg)

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

def nudge_eig(mat: jnp.ndarray,
              cond: float | None = None,
              min_eig: float | None  = None) -> jnp.ndarray:
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
    # use current constants

    EIG_COND = 1e-9
    EIG_MIN = 1e-12
    
    if not cond:
        cond = EIG_COND
    if not min_eig:
        min_eig = EIG_MIN

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
    \begin{equation}
    \exp\left\{\ln{\Gamma(N+1)} - [\ln{\Gamma(k+1)} + \ln{\Gamma(N+1-k)}]\right\}
    \end{equation}    
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

# Aliases, lookups and partials
conv_inter = convolve_interleaved
dB20 = mag_2_db
l1_norm = partial(jnp.linalg.norm, ord=1)
l1_norm_ax0 = partial(jnp.linalg.norm, ord=1, axis=0)
l1_norm_ax1 = partial(jnp.linalg.norm, ord=1, axis=1)
l2_norm = partial(jnp.linalg.norm, ord=2)
l2_norm_ax0 = partial(jnp.linalg.norm, ord=2, axis=0)
l2_norm_ax1 = partial(jnp.linalg.norm, ord=2, axis=1)
conv_cost = lambda x: dB20(l2_norm_ax0(conv_inter(l2_norm_ax0(x))))
    
FUNC_LOOKUP: dict[str, tuple[str, Callable | None]] = {
    're': ('Real Part', jnp.real),
    'im': ('Imag Part', jnp.imag),
    'mag': ('Magnitude', jnp.abs),
    'db': ('Magnitude (dB)', complex_2_db),
    'db10': ('Magnitude (dB)', complex_2_db10),
    'rad': ('Phase (rad)', jnp.angle),
    'deg': ('Phase (deg)', lambda x: jnp.angle(x, deg=True)),
    'arcl': ('Arc Length',lambda x: jnp.angle(x) * jnp.abs(x)),
    'rad_unwrap': ('Phase (rad)', lambda x: unwrap_rad(jnp.angle(x))),
    'deg_unwrap': ('Phase (deg)', lambda x: radian_2_degree(unwrap_rad(jnp.angle(x)))),
    'arcl_unwrap': ('Arc Length', lambda x: unwrap_rad(jnp.angle(x)) * jnp.abs(x)),
    'vswr': ('VSWR', lambda x: (1 + jnp.abs(x)) / (1 - jnp.abs(x))),
    # 'time': ('Time (real)', mf.ifft),
    # 'time_db': ('Magnitude (dB)',  lambda x: mf.complex_2_db(mf.ifft(x))),
    # 'time_mag': ('Magnitude', lambda x: mf.complex_2_magnitude(mf.ifft(x))),
}