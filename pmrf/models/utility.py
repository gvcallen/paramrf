import skrf

import jax.numpy as jnp
from pmrf._util import field
from pmrf.frequency import Frequency
from pmrf.models.model import Model

class Measured(Model):
    """
    A model wrapping a static Measured Network (e.g., from a Touchstone file).

    This model takes a `skrf.Network` and interpolates its S-parameters to the
    frequency grid requested during simulation.

    Attributes
    ----------
    network : skrf.Network
        The static network data containing S-parameters and frequency information.
        Marked as static to avoid tracing overhead in JAX.
    """
    network: skrf.Network = field(static=True)

    def s(self, freq: Frequency) -> jnp.ndarray:
        S_old = jnp.array(self.network.s)
        f_old = jnp.array(self.network.f)
        f_new = freq.f
        n_ports = S_old.shape[1]

        # Split into real and imaginary parts
        S_real = jnp.real(S_old)
        S_imag = jnp.imag(S_old)

        # Interpolate each real/imag component independently
        def interp_component(S_comp):
            return jnp.stack([
                jnp.stack([
                    jnp.interp(f_new, f_old, S_comp[:, i, j], left=jnp.nan, right=jnp.nan)
                    for j in range(n_ports)
                ], axis=0)
                for i in range(n_ports)
            ], axis=0)  # shape: (n_ports, n_ports, n_freqs_new)

        S_real_new = interp_component(S_real)
        S_imag_new = interp_component(S_imag)

        # Combine and transpose back to (n_freqs_new, n_ports, n_ports)
        S_new = (S_real_new + 1j * S_imag_new).transpose(2, 0, 1)

        return S_new

class SModel(Model):
    """
    A general model defined by a constant S-parameter matrix.

    Attributes
    ----------
    s_array : jnp.array
        The static S-parameter array.
    """
    s_array: jnp.array
    
    def s(self, _freq: Frequency) -> jnp.ndarray:
        nports = self.s_array.shape[1]
        nfreq = _freq.npoints

        if nfreq != self.s_array.shape[0]:
            return jnp.zeros((nfreq, nports, nports))

        return self.s_array
    

class AModel(Model):
    """
    A general model defined by a constant ABCD matrix.

    Attributes
    ----------
    a_array : jnp.array
        The static ABCD-parameter array.
    """
    a_array: jnp.array
    
    def a(self, _freq: Frequency) -> jnp.ndarray:
        nports = self.a_array.shape[1]
        nfreq = _freq.npoints
        if nfreq != self.a_array.shape[0]:
            return jnp.zeros((nfreq, nports, nports))
        
        return self.a_array
    
class ListModel(Model):
    """
    A container model that holds a list of sub-models.

    Attributes
    ----------
    models : list[Model]
        The list of child models.
    """
    models: list[Model]

class DictModel(Model):
    """
    A container model that holds a dictionary of sub-models.

    Attributes
    ----------
    models : dict[str, Model]
        A dictionary mapping names to child models.
    """
    models: dict[str, Model]

    def __post_init__(self):
        """
        Automatically sets the dictionary items as attributes of the instance.
        """
        for key, value in self.models:
            setattr(self, key, value)