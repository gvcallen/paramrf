import jax
import jax.numpy as jnp

from pmrf.parameters import Parameter
from pmrf.frequency import Frequency
from pmrf.models.model import Model

class PiCLC(Model):
    """
    **Overview**

    A 2-port model of a Pi-network with a Capacitor-Inductor-Capacitor topology.

    This model consists of a shunt capacitor (`C1`), a series inductor (`L`),
    and a second shunt capacitor (`C2`). It is a fundamental building block
    for various filters and matching networks, and is also commonly used to
    model the parasitic effects of physical components like SMD resistors.

    To ensure numerical stability when using with JAX, this model provides a
    special case for when the series inductance `L` is zero, where the network
    behaves as a single shunt capacitor.

    **Example**

    ```python
    import pmrf as prf

    # Create a Pi-network model, for example to represent parasitics
    parasitics = prf.models.PiCLC(
        C1=0.05e-12, # 50 fF
        L=0.1e-9,    # 100 pH
        C2=0.05e-12  # 50 fF
    )

    # Calculate its response over a frequency range
    freq = prf.Frequency(start=1, stop=20, npoints=201, unit='ghz')
    s_params = parasitics.s(freq)

    print(f"S21 at 10 GHz: {abs(s_params[freq.center_idx, 1, 0]):.2f}")
    ```
    """
    C1: Parameter = 1.0e-12
    L: Parameter = 1.0e-9
    C2: Parameter = 1.0e-12
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the ABCD-matrix of the Pi-network.

        This method dynamically chooses an appropriate calculation based on whether
        the inductance `L` is zero, making it safe for JAX-based optimization.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The resultant ABCD-matrix.
        """
        # This conditional dispatch is used to avoid division-by-zero errors
        # during JAX transformations if L becomes zero.
        return jax.lax.cond(
            self.L == 0.0,
            lambda: self.a_zero_inductance(freq),
            lambda: self.a_general(freq),
        )

    def a_general(self, freq: Frequency):
        # Internal method for the general case where L is non-zero.
        C1, C2, L = self.C1, self.C2, self.L
        w = freq.w
        Y1 = 1j * w * C1
        Y2 = 1j * w * C2
        Y3 = 1 / (1j * w * L)

        return jnp.array([
            [1 + Y2 / Y3,           1 / Y3          ],
            [Y1 + Y2 + Y1*Y2/Y3,    1 + Y1 / Y3     ],
        ]).transpose(2, 0, 1)

    def a_zero_inductance(self, freq: Frequency):
        # Internal method for the special case where L is zero.
        # The network simplifies to a single shunt capacitor C = C1 + C2.
        C1, C2 = self.C1, self.C2
        w = freq.w
        
        C = C1 + C2
        Y = 1j * w * C
        ones = jnp.ones_like(Y)
        zeros = jnp.zeros_like(Y)
        
        return jnp.array([
            [ones,  zeros],
            [Y,     ones]
        ]).transpose(2, 0, 1)