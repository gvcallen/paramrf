"""
Adapter models that store static RF network data.
"""

import skrf
import jax.numpy as jnp
import parax as prx

from pmrf.models import Model, Frequency
from pmrf.network_collection import NetworkCollection

class Measured(Model):
    """
    A model wrapping a static :class:`skrf.Network` or :class:`NetworkCollection`.

    This model takes a `skrf.Network` and interpolates its S-parameters to the
    frequency grid requested during simulation.

    Attributes
    ----------
    network : skrf.Network
        The static network data containing S-parameters and frequency information.
        Marked as static to avoid tracing overhead in JAX.
    """
    data: skrf.Network | NetworkCollection = prx.constrained(static=True)
    
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
        
        S_old = jnp.array(self.data.s)
        f_old = jnp.array(self.data.f)
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