import jax
import jax.numpy as jnp

from pmrf.parameters import Parameter
from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.functions import y2s

class PiCLC(Model):
    """
    A 2-port or 3-port model of a Pi-network with a Capacitor-Inductor-Capacitor topology.

    This model consists of a shunt capacitor (`C1`), a series inductor (`L`),
    and a second shunt capacitor (`C2`). It is a fundamental building block
    for various filters and matching networks, and is also commonly used to
    model the parasitic effects of physical components like SMD resistors.

    The parameter `three_port` determines whether all four ports are exposed or not.

    To ensure numerical stability when using with JAX, this model provides a
    special case for when the series inductance `L` is zero, where the network
    behaves as a single shunt capacitor.
    """
    C1: Parameter = 1.0e-12
    L: Parameter = 1.0e-9
    C2: Parameter = 1.0e-12
    three_port: bool = False

    def y(self, freq: Frequency) -> jnp.ndarray:
        if not self.three_port:
            raise Exception('y only available for pi-CLC for three_port == True')
        
        Y1 = 1j * freq.w * self.C1
        Y2 = 1j * freq.w * self.C2
        Y3 = 1 / (1j * freq.w * self.L)

        return jnp.array([
            [Y1 + Y3,       -Y3,            -Y1],
            [-Y3,           Y2 + Y3,        -Y2],
            [-Y1,           -Y2,            Y1 + Y2],
        ]).transpose(2, 0, 1)        
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        if self.three_port:
            raise Exception('Cannot calculate ABCD matrix of pi network when three_port == True')

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
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        if not self.three_port:
            return super().s(freq)
        
        return y2s(self.y(freq), self.z0)
    
class BoxCLCC(Model):
    """
    A 3-port or 4-port model of a Box-network with a Capacitor-Inductor-Capacitor-Capacitor topology.

    This model consists of a shunt capacitor (`C1`), a series inductor (`L`),
    a second shunt capacitor (`C2`), and a bridging capacitor (`C3`).

    The parameter `four_port` determines whether all four ports are exposed or not.
    """    
    C1: Parameter = 1.0e-12
    L: Parameter = 1.0e-9
    C2: Parameter = 1.0e-12
    C3: Parameter = 1.0e-12
    four_port: bool = False

    def y(self, freq: Frequency) -> jnp.ndarray:
        if not self.four_port:
            raise Exception('y only available for pi-CLC for four_port == True')

        return jax.lax.cond(
            jnp.array(self.L) <= 1e-18,
            lambda: self.y_zero_inductance(freq),
            lambda: self.y_general(freq),
        )
    
    def y_general(self, freq: Frequency) -> jnp.ndarray:        
        Y1 = 1j * freq.w * self.C1
        Y2 = 1j * freq.w * self.C2
        Y3 = 1 / (1j * freq.w * self.L)
        Y4 = 1j * freq.w * self.C3
        zero = jnp.zeros(freq.npoints)
        zero = zero.astype(dtype=complex)

        return jnp.array([
            [Y1 + Y3,       -Y3,            -Y1,            zero],
            [-Y3,           Y2 + Y3,        zero,              -Y2],
            [-Y1,           zero,              Y1 + Y4,        -Y4],
            [zero,             -Y2,            -Y4,            Y2 + Y4]
        ]).transpose(2, 0, 1)
    
    def y_zero_inductance(self, freq: Frequency) -> jnp.ndarray:        
        Y1 = 1j * freq.w * self.C1
        Y2 = 1j * freq.w * self.C2

        # Hack for now
        L = 1e-18
        Y3 = 1 / (1j * freq.w * L)
        Y4 = 1j * freq.w * self.C3
        zero = jnp.zeros(freq.npoints)
        zero = zero.astype(dtype=complex)

        return jnp.array([
            [Y1 + Y3,       -Y3,            -Y1,            zero],
            [-Y3,           Y2 + Y3,        zero,              -Y2],
            [-Y1,           zero,              Y1 + Y4,        -Y4],
            [zero,             -Y2,            -Y4,            Y2 + Y4]
        ]).transpose(2, 0, 1)     
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        if not self.four_port:
            raise Exception('Box-CLCC network not yet available for four_port == False')
        
        return y2s(self.y(freq), self.z0)