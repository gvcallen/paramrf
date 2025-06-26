from typing import Callable
from skrf.constants import INF, LOG_OF_NEG, NumberLike

import pmrf.numpy as np
from pmrf.numpy import USE_JAX
if USE_JAX:
    import jax.scipy as scipy
else:
    import scipy
from pmrf.numpy import USE_JAX, imag, pi, real, unwrap
from pmrf._misc import NumberLike

if USE_JAX:
    import jax
    from jax import lax
    from jax.scipy.special import gammaln
    from jax._src.numpy.ufuncs import _constant_like
else:
    from scipy.special import gammaln

def complex_2_magnitude(z: NumberLike):
    """
    Return the magnitude of the complex argument.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag : ndarray or scalar

    """
    return np.abs(z)


def complex_2_db(z: NumberLike):
    r"""
    Return the magnitude in dB of a complex number (as :math:`20\log_{10}(|z|)`)..

    The magnitude in dB is defined as :math:`20\log_{10}(|z|)`
    where :math:`z` is a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag20dB : ndarray or scalar
    """
    return magnitude_2_db(np.abs(z))


def complex_2_db10(z: NumberLike):
    r"""
    Return the magnitude in dB of a complex number (as :math:`10\log_{10}(|z|)`).

    The magnitude in dB is defined as :math:`10\log_{10}(|z|)`
    where :math:`z` is a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag10dB : ndarray or scalar
    """
    return mag_2_db10(np.abs(z))


def complex_2_radian(z: NumberLike):
    """
    Return the angle complex argument in radian.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    ang_rad : ndarray or scalar
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.
    """
    return np.angle(z)


def complex_2_degree(z: NumberLike):
    """
    Returns the angle complex argument in degree.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    ang_deg : ndarray or scalar
    """
    return np.angle(z, deg=True)


def complex_2_quadrature(z: NumberLike):
    r"""
    Take a complex number and returns quadrature, which is (length, arc-length from real axis)

    Arc-length is calculated as :math:`|z| \arg(z)`.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag : array like or scalar
        magnitude (length)
    arc_length : array like or scalar
        arc-length from real axis: angle*magnitude
    """
    return (np.abs(z), np.angle(z)*np.abs(z))


def complex_2_reim(z: NumberLike):
    """
    Return real and imaginary parts of a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    real : array like or scalar
        real part of input
    imag : array like or scalar
        imaginary part of input
    """
    return (np.real(z), np.imag(z))


def complex_components(z: NumberLike):
    """
    Break up a complex array into all possible scalar components.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    c_real : array like or scalar
        real part
    c_imag : array like or scalar
        imaginary part
    c_angle : array like or scalar
        angle in degrees
    c_mag : array like or scalar
        magnitude
    c_arc : array like or scalar
        arclength from real axis, angle*magnitude
    """
    return (*complex_2_reim(z), np.angle(z,deg=True), *complex_2_quadrature(z))


def magnitude_2_db(z: NumberLike, zero_nan: bool = True):
    """
    Convert linear magnitude to dB.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers
    zero_nan : bool, optional
        Replace NaN with zero. The default is True.

    Returns
    -------
    z : number or array_like
       Magnitude in dB given by 20*log10(|z|)
    """
    out = 20 * np.log10(z)
    if zero_nan:
        return np.nan_to_num(out, nan=LOG_OF_NEG, neginf=-np.inf)
    return out

mag_2_db = magnitude_2_db


def mag_2_db10(z: NumberLike, zero_nan:bool = True):
    """
    Convert linear magnitude to dB.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers
    zero_nan : bool, optional
        Replace NaN with zero. The default is True.

    Returns
    -------
    z : array_like
       Magnitude in dB given by 10*log10(|z|)
    """
    out = 10 * np.log10(z)
    if zero_nan:
        return np.nan_to_num(out, nan=LOG_OF_NEG, neginf=-np.inf)
    return out


def db_2_magnitude(z: NumberLike):
    """
    Convert dB to linear magnitude.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    z : number or array_like
        10**((z)/20) where z is a complex number
    """
    return 10**((z)/20.)

db_2_mag = db_2_magnitude


def db10_2_mag(z: NumberLike):
    """
    Convert dB to linear magnitude.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    z : array_like
        10**((z)/10) where z is a complex number
    """
    return 10**((z)/10.)


def magdeg_2_reim(mag: NumberLike, deg: NumberLike):
    """
    Convert linear magnitude and phase (in deg) arrays into a complex array.

    Parameters
    ----------
    mag : number or array_like
        A complex number or sequence of real numbers
    deg : number or array_like
        A complex number or sequence of real numbers

    Returns
    -------
    z : array_like
        A complex number or sequence of complex numbers

    """
    return mag*np.exp(1j*deg*pi/180.)

def dbdeg_2_reim(db: NumberLike, deg: NumberLike):
    """
    Converts dB magnitude and phase (in deg) arrays into a complex array.

    Parameters
    ----------
    db : number or array_like
        A realnumber or sequence of real numbers
    deg : number or array_like
        A real number or sequence of real numbers

    Returns
    -------
    z : array_like
        A complex number or sequence of complex numbers
    """
    return magdeg_2_reim(db_2_magnitude(db), deg)


def db_2_np(db: NumberLike):
    """
    Converts a value in decibel (dB) to neper (Np).

    Parameters
    ----------
    db : number or array_like
        A real number or sequence of real numbers

    Returns
    -------
    np : number or array_like
        A real number of sequence of real numbers
    """
    return (np.log(10)/20) * db


def np_2_db(x: NumberLike):
    """
    Converts a value in Nepers (Np) to decibel (dB).

    Parameters
    ----------
    np : number or array_like
        A real number or sequence of real numbers

    Returns
    -------
    db : number or array_like
        A real number of sequence of real numbers
    """
    return 20/np.log(10) * x


def radian_2_degree(rad: NumberLike):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    rad : number or array_like
        Angle in radian

    Returns
    -------
    deg : number or array_like
        Angle in degree
    """
    return (rad)*180/pi


def degree_2_radian(deg: NumberLike):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    deg : number or array_like
        Angle in radian

    Returns
    -------
    rad : number or array_like
        Angle in degree
    """
    return (deg)*pi/180.


def feet_2_meter(feet: NumberLike = 1):
    """
    Convert length in feet to meter.

    1 foot is equal to 0.3048 meters.

    Parameters
    ----------
    feet : number or array-like, optional
        length in feet. Default is 1.

    Returns
    -------
    meter: number or array-like
        length in meter

    See Also
    --------
    meter_2_feet
    """
    return 0.3048*feet

def meter_2_feet(meter: NumberLike = 1):
    """
    Convert length in meter to feet.

    1 meter is equal to 0.3.28084 feet.

    Parameters
    ----------
    meter : number or array-like, optional
        length in meter. Default is 1.

    Returns
    -------
    feet : number or array-like
        length in feet

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
    db_per_100feet : number or array-like, optional
        Attenuation in dB/ 100 ft. Default is 1.

    Returns
    -------
    db_per_100meter : number or array-like
        Attenuation in dB/ 100 m

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
    Unwraps a phase given in radians.

    Parameters
    ----------
    phi : number or array_like
        phase in radians

    Returns
    -------
    phi : number of array_like
        unwrapped phase in radians
    """
    return unwrap(phi, axis=0)


def sqrt_known_sign(z_squared: NumberLike, z_approx: NumberLike):
    """
    Return the square root of a complex number, with sign chosen to match `z_approx`.

    Parameters
    ----------
    z_squared : number or array-like
        the complex to be square-rooted
    z_approx : number or array-like
        the approximate value of z. sign of z is chosen to match that of
        z_approx

    Returns
    -------
    z : number, array-like (same type as z_squared)
        square root of z_squared.
    """
    z = np.sqrt(z_squared)
    return np.where(
        np.sign(np.angle(z)) == np.sign(np.angle(z_approx)),
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
        root 1
    z2 : array-like
        root 2
    z_approx : array-like
        approximate answer of z

    Returns
    -------
    z3 : np.array
        array built from z1 and z2 by
        z1 where sign(z1) == sign(z_approx), z2 else

    """
    return np.where(
    np.sign(np.angle(z1)) == np.sign(np.angle(z_approx)),z1, z2)


def find_closest(z1: NumberLike, z2: NumberLike, z_approx: NumberLike):
    """
    Return z1 or z2  depending on which is  closer to z_approx.

    Parameters
    ----------
    z1 : array-like
        root 1
    z2 : array-like
        root 2
    z_approx : array-like
        approximate answer of z

    Returns
    -------
    z3 : np.array
        array built from z1 and z2

    """
    z1_dist = abs(z1-z_approx)
    z2_dist = abs(z2-z_approx)

    return np.where(z1_dist<z2_dist,z1, z2)

def sqrt_phase_unwrap(z: NumberLike):
    r"""
    Take the square root of a complex number with unwrapped phase.

    This idea came from Lihan Chen.

    .. math::

        \sqrt{|z|} \exp( \arg_{unwrap}(z) / 2 )


    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    z : number of array_like
        A complex number or sequence of complex numbers
    """
    return np.sqrt(abs(z))*\
            np.exp(0.5*1j*unwrap_rad(complex_2_radian(z)))


# mathematical functions
def dirac_delta(x: NumberLike):
    r"""
    Calculate Dirac function.

    Dirac function :math:`\delta(x)` defined as :math:`\delta(x)=1` if x=0,
    0 otherwise.

    Parameters
    ----------
    x : number of array_like
        A real number or sequence of real numbers

    Returns
    -------
    delta : number of array_like
        1 or 0

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
        A real number or sequence of real numbers

    Returns
    -------
    y : number or array_like
        A real number or sequence of real numbers

    See Also
    --------
    dirac_delta
    """
    return 2. - dirac_delta(x)


def null(A: np.ndarray, eps: float = 1e-15):
    """
    Calculate the null space of matrix A.

    Parameters
    ----------
    A : array_like
    eps : float

    Returns
    -------
    null_space : array_like

    References
    ----------
    https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
    """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def inf_to_num(x: NumberLike):
    """
    Convert inf and -inf's to large numbers.

    Parameters
    ------------
    x : array-like or number
        the input array or number

    Returns
    -------
    x : Number of array_like
        Input without with +/- inf replaced by large numbers
    """
    x = np.nan_to_num(x, nan=np.nan, posinf=INF, neginf=-1*INF)
    return x


def cross_ratio(a: NumberLike, b: NumberLike, c: NumberLike, d:NumberLike):
    r"""
    Calculate the cross ratio of a quadruple of distinct points on the real line.


    The cross ratio is defined as:


    .. math::

        r = \frac{ (a-b)(c-d) }{ (a-d)(c-b) }



    Parameters
    ----------
    a,b,c,d :  array-like or number

    Returns
    -------
    r : array-like or number

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
        function of real variable
    name : string, optional
        name of the real/imag argument names if they are not first

    Returns
    -------
    f_c : Callable
        function of a complex variable

    Examples
    --------
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
    Serialize a list/array of complex numbers

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    re_im : array_like
        produce the following array for an input z:
        z[0].real, z[0].imag, z[1].real, z[1].imag, etc.

    See Also
    --------
    scalar2Complex
    """
    z = np.array(z)
    re_im = []
    for k in z:
        re_im.append(np.real(k))
        re_im.append(np.imag(k))
    return np.array(re_im).flatten()

def scalar2Complex(s: NumberLike):
    """
    Unserialize a list/array of real and imag numbers into a complex array.

    Inverse of :func:`complex2Scalar`.

    Parameters
    ----------
    s : array_like
        an array with real and imaginary parts ordered as:
        z[0].real, z[0].imag, z[1].real, z[1].imag, etc.

    Returns
    -------
    z : Number or array_like
        A complex number or sequence of complex number

    See Also
    --------
    complex2Scalar
    """
    s = np.array(s)
    z = []

    for k in range(0,len(s),2):
        z.append(s[k] + 1j*s[k+1])
    return np.array(z).flatten()    

def multiply_by(x: np.ndarray, by: np.ndarray, axis=None) -> np.ndarray:
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
            by = np.tile(by, (int(x.shape[0] / n), x.shape[1]))
        elif axis == 1:
            by = np.tile(by, (x.shape[0], int(x.shape[1] / n)))

        x *= by

    return x

def sum_every(x: np.ndarray, n: int, axis=None) -> np.ndarray:
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

def multiply_every(x: np.ndarray, n: int, axis=None) -> np.ndarray:
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

def L1(x, axis=0) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, ord=1)

def L2(x, axis=0) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, ord=2)

def convolve_interleaved(x, axis=1) -> np.ndarray:
    if axis == 1:
        if len(x.shape) == 1:
            y1, y2 = x[0::2], x[1::2]
            x = np.convolve(y1, y2)
        else:
            raise Exception("Not yet implemented")
    else:
        raise Exception("Not yet implemented")

    return x

def rms(x):
    return np.sqrt(np.mean(x**2))

def mag_2_db(x):
    return 20 * np.log10(np.abs(x))

def dB20(x):
    return mag_2_db(x)

def db_2_mag(x):
    return 10 ** (x / 20)

def polar_2_rect(radii, angles, deg=False):
    if deg:
        angles = np.deg2rad(angles)
    return radii * np.exp(1j*angles)

def rect_2_polar(x, deg=False):
    return abs(x), np.angle(x, deg=deg)

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