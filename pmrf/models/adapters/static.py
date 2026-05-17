"""
Models that store and interpolate static network data from raw arrays.
"""

import numpy as np
import jax.numpy as jnp

import skrf

from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection
from pmrf.models import Model
from pmrf.utils import field, freeze


def interpolate_network_data(f_old: jnp.ndarray, f_new: jnp.ndarray, data_old: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolates 3D network parameter data (Freq, Port, Port) across new frequencies.

    Parameters
    ----------
    f_old : jnp.ndarray
        The original frequency grid (1D array).
    f_new : jnp.ndarray
        The target frequency grid (1D array) requested for simulation.
    data_old : jnp.ndarray
        The network parameters to interpolate, shape (n_freqs, n_ports, n_ports).

    Returns
    -------
    jnp.ndarray
        The interpolated network parameters, shape (n_freqs_new, n_ports, n_ports).
    """
    # Ensure inputs are JAX arrays
    f_old = jnp.asarray(f_old)
    f_new = jnp.asarray(f_new)
    data_old = jnp.asarray(data_old)
    
    n_ports = data_old.shape[1]

    # Split into real and imaginary parts
    data_real = jnp.real(data_old)
    data_imag = jnp.imag(data_old)

    # Interpolate each real/imag component independently
    def interp_component(data_comp):
        return jnp.stack([
            jnp.stack([
                jnp.interp(f_new, f_old, data_comp[:, i, j], left=jnp.nan, right=jnp.nan)
                for j in range(n_ports)
            ], axis=0)
            for i in range(n_ports)
        ], axis=0)  # shape: (n_ports, n_ports, n_freqs_new)

    data_real_new = interp_component(data_real)
    data_imag_new = interp_component(data_imag)

    # Combine and transpose back to (n_freqs_new, n_ports, n_ports)
    return (data_real_new + 1j * data_imag_new).transpose(2, 0, 1)


class Measured(Model):
    """
    A model wrapping a static :class:`skrf.Network` or :class:`NetworkCollection`.

    This model takes a `skrf.Network` and interpolates its S-parameters to the
    frequency grid requested during simulation, utilizing a shared interpolation function.

    Parameters
    ----------
    data : skrf.Network | NetworkCollection
        The static network data containing S-parameters and frequency information.
    """
    #: The underlying network data.
    data: skrf.Network | NetworkCollection = field(static=True)
    
    def __getattr__(self, name: str) -> 'Measured':
        data = self.__getattribute__('data')

        if isinstance(data, NetworkCollection) and name in data.to_dict():
            return Measured(data[name])
        elif isinstance(data, skrf.Network) and name == data.name:
            return Measured(data)
        return super().__getattr__(name)
    
    def __post_init__(self):
        self.data.renormalize(self.z0, 'power')

    def s(self, freq: Frequency) -> jnp.ndarray:
        if isinstance(self.data, NetworkCollection):
            raise Exception("Cannot call s() on a Measured model that contains a NetworkCollection")
        
        # Delegate to the shared interpolation helper
        return interpolate_network_data(
            f_old=self.data.f, 
            f_new=freq.f, 
            data_old=self.data.s
        )


class SModel(Model):
    """
    A model storing static S-parameters (scattering) as raw arrays.

    Parameters
    ----------
    freq : Frequency
        The frequency object containing the grid of the static data.
    data : np.ndarray | jnp.ndarray
        The static S-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    """
    #: The frequency grid of the static data.
    freq: Frequency = field(converter=freeze)
    #: The static S-parameter matrix data.
    data: np.ndarray | jnp.ndarray = field(converter=freeze)

    def s(self, freq: Frequency) -> jnp.ndarray:
        return interpolate_network_data(self.freq.f, freq.f, self.data)


class AModel(Model):
    """
    A model storing static ABCD-parameters (cascade) as raw arrays.

    Parameters
    ----------
    freq : Frequency
        The frequency object containing the grid of the static data.
    data : np.ndarray | jnp.ndarray
        The static ABCD-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    """
    #: The frequency grid of the static data.
    freq: Frequency = field(converter=freeze)
    #: The static ABCD-parameter matrix data.
    data: np.ndarray | jnp.ndarray = field(converter=freeze)

    def a(self, freq: Frequency) -> jnp.ndarray:
        return interpolate_network_data(self.freq.f, freq.f, self.data)


class YModel(Model):
    """
    A model storing static Y-parameters (admittance) as raw arrays.

    Parameters
    ----------
    freq : Frequency
        The frequency object containing the grid of the static data.
    data : np.ndarray | jnp.ndarray
        The static Y-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    """
    #: The frequency grid of the static data.
    freq: Frequency = field(converter=freeze)
    #: The static Y-parameter matrix data.
    data: np.ndarray | jnp.ndarray = field(converter=freeze)

    def y(self, freq: Frequency) -> jnp.ndarray:
        return interpolate_network_data(self.freq.f, freq.f, self.data)


class ZModel(Model):
    """
    A model storing static Z-parameters (impedance) as raw arrays.

    Parameters
    ----------
    freq : Frequency
        The frequency object containing the grid of the static data.
    data : np.ndarray | jnp.ndarray
        The static Z-parameter matrix data of shape (n_freqs, n_ports, n_ports).
    """
    #: The frequency grid of the static data.
    freq: Frequency = field(converter=freeze)
    #: The static Z-parameter matrix data.
    data: np.ndarray | jnp.ndarray = field(converter=freeze)

    def z(self, freq: Frequency) -> jnp.ndarray:
        return interpolate_network_data(self.freq.f, freq.f, self.data)