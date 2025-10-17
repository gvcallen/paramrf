import jax
from jax import vmap
import skrf

import jax.numpy as jnp
from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.models.lumped import MATCH, SHORT
from pmrf._util import field

class Measured(Model):
    s_measured: jnp.ndarray
    f_measured: jnp.ndarray

    def __init__(self, ntwk: skrf.Network | str):
        if isinstance(ntwk, str):
            ntwk = skrf.Network(ntwk)

        self.s_measured = jnp.array(ntwk.s)
        self.f_measured = jnp.array(ntwk.f)

    def s(self, freq: Frequency) -> jnp.ndarray:
        S, f = self.s_measured, self.f_measured
        f_new = freq.f
        
        n_ports = S.shape[1]

        # Clip f_new to be within f bounds
        f_min, f_max = f[0], f[-1]
        f_new_clipped = jnp.clip(f_new, f_min, f_max)

        # Split into real and imaginary parts
        S_real = jnp.real(S)
        S_imag = jnp.imag(S)

        # Interpolate each real/imag component independently
        def interp_component(S_comp):
            return jnp.stack([
                jnp.stack([
                    jnp.interp(f_new_clipped, f, S_comp[:, i, j])
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