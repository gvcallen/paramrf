"""
Models that store and interpolate static network data from raw arrays.
"""

from typing import Literal

import numpy as np
import jax
import jax.numpy as jnp

import parax as prx
import skrf
from scipy.interpolate import CubicSpline

from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection
from pmrf.models import Model
from pmrf.utils import field, freeze
from pmrf.types import ArrayLike
from pmrf.rf import renormalize_s


def interpolate_network_data(f_old: jnp.ndarray, f_new: jnp.ndarray, data_old: jnp.ndarray) -> jnp.ndarray:
    r"""
    Linearly interpolate network data onto a new frequency grid.

    Every port pair shares one interval lookup and one gather, so the traced
    program has the same size at any port count (#130). Linear interpolation of
    a complex value is the same as interpolating its real and imaginary parts
    independently.

    **Mathematical Formulation**

    For $f_k \le f \le f_{k+1}$,

    $$X(f) = X_k + \frac{f - f_k}{f_{k+1} - f_k}\,(X_{k+1} - X_k),$$

    and $X(f)$ is NaN for $f$ outside $[f_0, f_{N-1}]$.

    Parameters
    ----------
    f_old : jnp.ndarray
        The increasing source frequencies, shape ``(N,)``.
    f_new : jnp.ndarray
        The requested frequencies, shape ``(M,)``.
    data_old : jnp.ndarray
        The source matrices, shape ``(N, n_ports, n_ports)``.

    Returns
    -------
    jnp.ndarray
        The interpolated matrices, shape ``(M, n_ports, n_ports)``.
    """
    f_old = jnp.asarray(f_old)
    f_new = jnp.asarray(f_new)
    data_old = jnp.asarray(data_old)

    interval = jnp.clip(
        jnp.searchsorted(f_old, f_new, side="right") - 1, 0, f_old.shape[0] - 2
    )
    f_lo = f_old[interval]
    weight = ((f_new - f_lo) / (f_old[interval + 1] - f_lo))[:, None, None]
    data_lo = data_old[interval]
    data_new = data_lo + weight * (data_old[interval + 1] - data_lo)

    outside_range = (f_new < f_old[0]) | (f_new > f_old[-1])
    nan = jnp.asarray(complex(np.nan, np.nan), dtype=data_new.dtype)
    return jnp.where(outside_range[:, None, None], nan, data_new)


def _cubic_spline_coefficients(
    f: np.ndarray, data: np.ndarray
) -> prx.Static[tuple[tuple[int, ...], str, bytes]]:
    """Precompute not-a-knot cubic spline coefficients for complex data."""
    f = np.asarray(f)
    data = np.asarray(data)
    f_normalized = (f - f[0]) / (f[-1] - f[0])

    real_coefficients = CubicSpline(
        f_normalized, np.real(data), axis=0
    ).c
    imag_coefficients = CubicSpline(
        f_normalized, np.imag(data), axis=0
    ).c
    coefficients = real_coefficients + 1j * imag_coefficients
    payload = (coefficients.shape, coefficients.dtype.str, coefficients.tobytes())
    return prx.Static(payload)


def _restore_cubic_spline_coefficients(
    payload: tuple[tuple[int, ...], str, bytes],
) -> np.ndarray:
    shape, dtype, buffer = payload
    return np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(shape)


def _interpolate_network_data_cubic(
    f_old: jnp.ndarray,
    f_new: jnp.ndarray,
    coefficients: np.ndarray,
) -> jnp.ndarray:
    """Evaluate precomputed cubic spline coefficients using JAX operations."""
    f_old = jnp.asarray(f_old)
    f_new = jnp.asarray(f_new)
    coefficients = jnp.asarray(coefficients)
    f_normalized = (f_old - f_old[0]) / (f_old[-1] - f_old[0])
    f_new_normalized = (f_new - f_old[0]) / (f_old[-1] - f_old[0])

    interval = jnp.searchsorted(f_normalized, f_new_normalized, side="right") - 1
    interval = jnp.clip(interval, 0, f_old.shape[0] - 2)
    offset = (f_new_normalized - f_normalized[interval])[:, None, None]

    selected = coefficients[:, interval, :, :]
    data_new = (
        (selected[0] * offset + selected[1]) * offset + selected[2]
    ) * offset + selected[3]
    outside_range = (f_new < f_old[0]) | (f_new > f_old[-1])
    complex_nan = jnp.asarray(complex(np.nan, np.nan), dtype=data_new.dtype)
    return jnp.where(outside_range[:, None, None], complex_nan, data_new)


def renormalize_network_data(s_old: jnp.ndarray, z0_old: jnp.ndarray, z0_new: jnp.ndarray) -> jnp.ndarray:
    z0_new_arr = jnp.asarray(z0_new)
    is_matched = jnp.all(z0_new_arr == z0_old)
    
    def _renorm():
        return renormalize_s(
            s=s_old, 
            z_old=z0_old, 
            z_new=z0_new_arr, 
            s_def_old='power', 
            s_def_new='power'
        )
        
    def _identity():
        return s_old
        
    return jax.lax.cond(is_matched, _identity, _renorm)


class _ByIdentity:
    """A static-field holder compared by identity.

    `eqx.filter_jit` compares static fields with `==`. `skrf.Network.__eq__`
    broadcasts the S-parameter arrays of both networks, which is O(data) on every
    call and raises for networks of different shapes. Identity is exact here,
    because a `SkrfNetwork` owns a private copy of its network.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _ByIdentity) and self.value is other.value

    def __hash__(self):
        return id(self.value)


class SkrfNetwork(Model):
    """
    A model wrapping a static :class:`skrf.Network` or :class:`NetworkCollection`.

    This model takes a `skrf.Network` and interpolates its S-parameters to the
    frequency grid requested during simulation.

    Parameters
    ----------
    network : skrf.Network | NetworkCollection
        The static network data containing S-parameters and frequency information.
    interpolation_kind : {"linear", "cubic"}, default="linear"
        Interpolation applied independently to the real and imaginary parts of
        the S-parameters.
    """
    #: The underlying network data, compared by identity in the jit cache key.
    _network: "_ByIdentity" = field(static=True, repr=False)

    #: The interpolation used when evaluating at a new frequency grid.
    interpolation_kind: Literal["linear", "cubic"] = field(static=True)

    _spline_coefficients: prx.Static[
        tuple[tuple[int, ...], str, bytes]
    ] | None = field(repr=False)

    def __init__(
        self,
        network: skrf.Network | NetworkCollection,
        interpolation_kind: Literal["linear", "cubic"] = "linear",
    ):
        if interpolation_kind not in ("linear", "cubic"):
            raise ValueError(
                "interpolation_kind must be either 'linear' or 'cubic', "
                f"got {interpolation_kind!r}"
            )
        self.interpolation_kind = interpolation_kind
        self._spline_coefficients = None

        if isinstance(network, skrf.Network):
            network = network.copy()
            network.renormalize(50.0, 'power')
            if interpolation_kind == "cubic":
                self._spline_coefficients = _cubic_spline_coefficients(
                    network.f, network.s
                )
        self._network = _ByIdentity(network)

    @property
    def network(self) -> skrf.Network | NetworkCollection:
        """The underlying network data, renormalised to 50 ohm."""
        return self._network.value

    def __getattr__(self, name: str) -> 'SkrfNetwork':
        if name.startswith('_'):
            return super().__getattr__(name)
        network = self.__getattribute__('network')

        if isinstance(network, NetworkCollection) and name in network.to_dict():
            return SkrfNetwork(
                network[name], interpolation_kind=self.interpolation_kind
            )
        elif isinstance(network, skrf.Network) and name == network.name:
            return SkrfNetwork(
                network, interpolation_kind=self.interpolation_kind
            )
        return super().__getattr__(name)
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        if isinstance(self.network, NetworkCollection):
            raise Exception("Cannot call s() on a Measured model that contains a NetworkCollection")
        
        f_old = jnp.array(self.network.f)
        s_old = jnp.array(self.network.s)
        
        if self.interpolation_kind == "linear":
            s_interp = interpolate_network_data(
                f_old=f_old,
                f_new=freq.f,
                data_old=s_old,
            )
        else:
            coefficients = self._spline_coefficients
            s_interp = _interpolate_network_data_cubic(
                f_old=f_old,
                f_new=freq.f,
                coefficients=_restore_cubic_spline_coefficients(coefficients),
            )
        
        return renormalize_network_data(s_interp, 50.0, z0)
        
        
class Touchstone(Model):
    """
    A model for a touchstone file.

    This internally uses :class:`pmrf.models.SkrfNetwork` to load the touchstone.

    Parameters
    ----------
    file : str
        The file to open.
    interpolation_kind : {"linear", "cubic"}, default="linear"
        The interpolation kind forwarded to :class:`SkrfNetwork`.
    """
    #: The underlying Network model use to encapsulate the touchstone.
    touchstone: SkrfNetwork = field(static=True)
    
    def __init__(
        self,
        file: str,
        interpolation_kind: Literal["linear", "cubic"] = "linear",
        **kwargs,
    ):
        skrf_network = SkrfNetwork(
            skrf.Network(file, **kwargs),
            interpolation_kind=interpolation_kind,
        )
        self.touchstone = skrf_network

    @property
    def interpolation_kind(self) -> Literal["linear", "cubic"]:
        return self.touchstone.interpolation_kind
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return self.touchstone.s(freq, z0=z0)


class SModel(Model):
    """
    A model storing static S-parameters (scattering) as raw arrays.

    Parameters
    ----------
    s_matrix : np.ndarray
        The static S-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    frequency : Frequency
        The frequency object containing the grid of the static data.
    z0 : np.ndarray
        The characteristic impedance for which `s_matrix` is defined.
        Can be initialized with any float-like value.
    """
    #: The static S-parameter matrix data.
    s_matrix: np.ndarray
    
    #: The frequency grid of the static data.
    frequency: Frequency = field(converter=freeze)
    
    #: The z0 for which the data is defined.
    z0: np.ndarray = field(converter=np.asarray)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s_matrix_interp = interpolate_network_data(self.frequency.f, freq.f, self.s_matrix)
        return renormalize_network_data(s_matrix_interp, self.z0, z0)
        


class AModel(Model):
    """
    A model storing static ABCD-parameters (cascade) as raw arrays.

    Parameters
    ----------
    a_matrix : np.ndarray
        The static ABCD-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    frequency : Frequency
        The frequency object containing the grid of the static data.
    """
    #: The static ABCD-parameter matrix data.
    a_matrix: np.ndarray
    
    #: The frequency grid of the static data.
    frequency: Frequency = field(converter=freeze)

    def a(self, freq: Frequency) -> jnp.ndarray:
        return interpolate_network_data(self.frequency.f, freq.f, self.a_matrix)


class YModel(Model):
    """
    A model storing static Y-parameters (admittance) as raw arrays.

    Parameters
    ----------
    y_matrix : np.ndarray
        The static Y-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    frequency : Frequency
        The frequency object containing the grid of the static data.
    """
    #: The static Y-parameter matrix data.
    y_matrix: np.ndarray
    
    #: The frequency grid of the static data.
    frequency: Frequency = field(converter=freeze)

    def y(self, freq: Frequency) -> jnp.ndarray:
        return interpolate_network_data(self.frequency.f, freq.f, self.y_matrix)


class ZModel(Model):
    """
    A model storing static Z-parameters (impedance) as raw arrays.

    Parameters
    ----------
    z_matrix : np.ndarray
        The static Z-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    frequency : Frequency
        The frequency object containing the grid of the static data.
    """
    #: The static Z-parameter matrix data.
    z_matrix: np.ndarray
    
    #: The frequency grid of the static data.
    frequency: Frequency = field(converter=freeze)

    def z(self, freq: Frequency) -> jnp.ndarray:
        return interpolate_network_data(self.frequency.f, freq.f, self.z_matrix)
