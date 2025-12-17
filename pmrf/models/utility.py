import skrf

import jax.numpy as jnp
from pmrf._util import field
from pmrf.frequency import Frequency
from pmrf.models.model import Model

class Measured(Model):
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
    
class ListModel(Model):
    models: list[Model]

class DictModel(Model):
    models: dict[str, Model]

    def __post_init__(self):
        for key, value in self.models:
            setattr(self, key, value)

class SModel(Model):
    """
    **Overview**

    A general model defined by a constant S-parameter matrix.
    """
    s_array: jnp.array
    
    def s(self, _freq: Frequency) -> jnp.ndarray:
        """Returns the S-parameter matrix.

        Args:
            freq (Frequency): Specifies the frequency to calculate the parameters at.

        Returns:
            jnp.ndarray: The resultant block-diagonal S-parameter matrix.
        """
        nports = self.s_array.shape[1]
        nfreq = _freq.npoints

        if nfreq != self.s_array.shape[0]:
            return jnp.zeros((nfreq, nports, nports))

        return self.s_array
    

class AModel(Model):
    """
    **Overview**

    A general model defined by a constant ABCD matrix.
    """
    a_array: jnp.array
    
    def a(self, _freq: Frequency) -> jnp.ndarray:
        """Returns the ABCD matrix.

        Args:
            freq (Frequency): Specifies the frequency to calculate the parameters at.

        Returns:
            jnp.ndarray: The resultant block-diagonal ABCD matrix.
        """
        nports = self.a_array.shape[1]
        nfreq = _freq.npoints
        if nfreq != self.a_array.shape[0]:
            return jnp.zeros((nfreq, nports, nports))
        
        return self.a_array