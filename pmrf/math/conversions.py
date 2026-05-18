"""
Math conversion functions.

A large portion of these function have been ported from scikit-rf.
"""
from typing import Callable

import jax.numpy as jnp
from jax.numpy import pi
from jaxtyping import ArrayLike

from pmrf.math.misc import unwrap_rad

LOG_OF_NEG = -100

def complex_2_magnitude(z: ArrayLike):
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


def complex_2_db(z: ArrayLike):
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


def complex_2_db10(z: ArrayLike):
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


def complex_2_radian(z: ArrayLike):
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


def complex_2_degree(z: ArrayLike):
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


def complex_2_quadrature(z: ArrayLike):
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


def complex_2_reim(z: ArrayLike):
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
    return jnp.stack([jnp.real(z), jnp.imag(z)])


def complex_components(z: ArrayLike):
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


def magnitude_2_db(z: ArrayLike, zero_nan: bool = True):
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
        Magnitude in dB20.
    """
    out = 20 * jnp.log10(jnp.abs(z))
    if zero_nan:
        return jnp.nan_to_num(out, nan=LOG_OF_NEG, neginf=-jnp.inf)
    return out

mag_2_db = magnitude_2_db


def mag_2_db10(z: ArrayLike, zero_nan:bool = True):
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
        Magnitude in dB10.
    """
    out = 10 * jnp.log10(jnp.abs(z))
    if zero_nan:
        return jnp.nan_to_num(out, nan=LOG_OF_NEG, neginf=-jnp.inf)
    return out


def db_2_magnitude(z: ArrayLike):
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


def db10_2_mag(z: ArrayLike):
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


def magdeg_2_reim(mag: ArrayLike, deg: ArrayLike):
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

def dbdeg_2_reim(db: ArrayLike, deg: ArrayLike):
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


def db_2_np(db: ArrayLike):
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


def np_2_db(x: ArrayLike):
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


def radian_2_degree(rad: ArrayLike):
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


def degree_2_radian(deg: ArrayLike):
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


def feet_2_meter(feet: ArrayLike = 1):
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

def meter_2_feet(meter: ArrayLike = 1):
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


def db_per_100feet_2_db_per_100meter(db_per_100feet: ArrayLike = 1):
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


def sqrt_phase_unwrap(z: ArrayLike):
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
            kw_re = {name: jnp.real(z)}
            kw_im = {name: jnp.imag(z)}
            kw_re.update(kw)
            kw_im.update(kw)
            return f(*args, **kw_re) + 1j*f(*args, **kw_im)
        else:
            return f(jnp.real(z), *args,**kw) + 1j*f(jnp.imag(z), *args, **kw)
    return f_c


CONVERSION_LOOKUP: dict[str, tuple[str, Callable | None]] = {
    're': ('Real Part', jnp.real),
    'im': ('Imag Part', jnp.imag),
    'abs': ('Magnitude', jnp.abs),
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
