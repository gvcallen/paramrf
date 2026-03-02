"""
Adapter models that store static RF network data.
"""

import skrf
import jax.numpy as jnp

from pmrf.models.adapters.base import Model
from pmrf.frequency import Frequency
from pmrf.util import field

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