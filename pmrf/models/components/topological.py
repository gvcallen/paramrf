"""
Specific topology layouts such as Pi-CLC or Box-CLCC networks.
"""
import jax
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.rf import y2s
from pmrf.parameters import Param, param
from pmrf.constraints import Positive

class PiCLC(Model):
    """
    A 2-port model of a Pi-network with a Capacitor-Inductor-Capacitor topology.

    This model consists of a shunt capacitor (`C1`), a series inductor (`L`),
    and a second shunt capacitor (`C2`). It is a fundamental building block
    for various filters and matching networks, and is also commonly used to
    model the parasitic effects of physical components like SMD resistors.

    Parameters
    ----------
    C1 : Param
        The value of the first shunt capacitor in Farads.
    L : Param
        The value of the series inductor in Henrys.
    C2 : Param
        The value of the second shunt capacitor in Farads.
        If True, treats the network as a 3-port device (where the ground reference is implicit or shared).
        If False, treats it as a standard 2-port network.
    """
    #: The value of the first shunt capacitor in Farads.
    C1: Param = param(constraint=Positive())
    
    #: The value of the series inductor in Henrys.
    L: Param = param(constraint=Positive())
    
    #: The value of the second shunt capacitor in Farads.
    C2: Param = param(constraint=Positive())

    # def y(self, freq: Frequency) -> jnp.ndarray:
    #     if not self.three_port:
    #         raise Exception('y only available for pi-CLC for three_port == True')
        
    #     Y1 = 1j * freq.w * self.C1
    #     Y2 = 1j * freq.w * self.C2
    #     Y3 = 1 / (1j * freq.w * self.L)

    #     return jnp.array([
    #         [Y1 + Y3,       -Y3,            -Y1],
    #         [-Y3,           Y2 + Y3,        -Y2],
    #         [-Y1,           -Y2,            Y1 + Y2],
    #     ]).transpose(2, 0, 1)        
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        return jax.lax.cond(
            self.L == 0.0,
            lambda: self.a_zero_inductance(freq),
            lambda: self.a_general(freq),
        )

    def a_general(self, freq: Frequency):
        """
        Internal calculation for the general case (L != 0).

        Parameters
        ----------
        freq : Frequency
            The frequency points.

        Returns
        -------
        jnp.ndarray
            The ABCD matrix.
        """
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
        """
        Internal calculation for the zero inductance case (L == 0).

        The network simplifies to a single shunt capacitor C = C1 + C2.

        Parameters
        ----------
        freq : Frequency
            The frequency points.

        Returns
        -------
        jnp.ndarray
            The ABCD matrix.
        """
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
    
    # def s(self, freq: Frequency) -> jnp.ndarray:
    #     if not self.three_port:
    #         from pmrf.rf import a2s
    #         return a2s(self.a(freq), self.z0)
        
    #     return y2s(self.y(freq), self.z0)
    
class BoxCLCC(Model):
    """
    A 4-port model of a Box-network with a Capacitor-Inductor-Capacitor-Capacitor topology.

    This model consists of a shunt capacitor (`C1`), a series inductor (`L`),
    and a second shunt capacitor (`C2`), and a bridging capacitor (`C3`).

    The parameter `four_port` determines whether all four ports are exposed or not.

    Parameters
    ----------
    C1 : Param
        First shunt capacitor.
    L : Param
        Series inductor.
    C2 : Param
        Second shunt capacitor.
    C3 : Param
        Bridging capacitor.
    """    
    #: First shunt capacitor.
    C1: Param = param(constraint=Positive())
    
    #: Series inductor.
    L: Param = param(constraint=Positive())
    
    #: Second shunt capacitor.
    C2: Param = param(constraint=Positive())
    
    #: Bridging capacitor.
    C3: Param = param(constraint=Positive())

    def y(self, freq: Frequency) -> jnp.ndarray:
        return jax.lax.cond(
            jnp.array(self.L) <= 1e-18,
            lambda: self.y_zero_inductance(freq),
            lambda: self.y_general(freq),
        )
    
    def y_general(self, freq: Frequency) -> jnp.ndarray:        
        """
        Internal calculation for the general case (L > 1e-18).

        Parameters
        ----------
        freq : Frequency
            The frequency points.

        Returns
        -------
        jnp.ndarray
            The Y matrix.
        """
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
        """
        Internal calculation for the zero inductance case.

        Uses a small epsilon for L to avoid division by zero while approximating
        the behavior.

        Parameters
        ----------
        freq : Frequency
            The frequency points.

        Returns
        -------
        jnp.ndarray
            The Y matrix.
        """
        Y1 = 1j * freq.w * self.C1
        Y2 = 1j * freq.w * self.C2

        # TODO Currently we re-formulate in terms of a small inductance which will work
        # for any application, though we should try re-write the maths rigourously.
        L = 1e-30
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
        return y2s(self.y(freq), self.z0)
    

class TeeLCL(Model):
    """
    A 2-port model of a Tee-network with an Inductor-Capacitor-Inductor topology.

    This is the dual of the Pi-CLC network. It consists of a series inductor (`L1`),
    a shunt capacitor (`C`), and a second series inductor (`L2`). It is often 
    used for low-pass filtering and impedance matching.

    Parameters
    ----------
    L1 : Param
        The value of the first series inductor in Henrys.
    C : Param
        The value of the shunt capacitor in Farads.
    L2 : Param
        The value of the second series inductor in Henrys.
    """
    #: The value of the first series inductor in Henrys.
    L1: Param = param(constraint=Positive())
    
    #: The value of the shunt capacitor in Farads.
    C: Param = param(constraint=Positive())
    
    #: The value of the second series inductor in Henrys.
    L2: Param = param(constraint=Positive())

    def a(self, freq: Frequency) -> jnp.ndarray:
        return jax.lax.cond(
            self.C == 0.0,
            lambda: self.a_zero_capacitance(freq),
            lambda: self.a_general(freq),
        )

    def a_general(self, freq: Frequency):
        """
        Internal calculation for the general case (C != 0).
        """
        L1, C, L2 = self.L1, self.C, self.L2
        w = freq.w
        Z1 = 1j * w * L1
        Z2 = 1j * w * L2
        Y3 = 1j * w * C

        return jnp.array([
            [1 + Z1 * Y3,           Z1 + Z2 + Z1 * Z2 * Y3],
            [Y3,                    1 + Z2 * Y3           ],
        ]).transpose(2, 0, 1)

    def a_zero_capacitance(self, freq: Frequency):
        """
        Internal calculation for the zero capacitance case (C == 0).

        The network simplifies to a single series inductor L = L1 + L2.
        """
        L1, L2 = self.L1, self.L2
        w = freq.w
        
        Z = 1j * w * (L1 + L2)
        ones = jnp.ones_like(Z)
        zeros = jnp.zeros_like(Z)
        
        return jnp.array([
            [ones,  Z],
            [zeros, ones]
        ]).transpose(2, 0, 1)

    def s(self, freq: Frequency) -> jnp.ndarray:
        from pmrf.rf import a2s
        return a2s(self.a(freq), self.z0)


class LSectionLC(Model):
    """
    A 2-port model of an L-section impedance matching network.
    
    This specific topology uses a series inductor (`L`) followed by a shunt 
    capacitor (`C`), acting as a standard low-pass impedance transformer.

    Parameters
    ----------
    L : Param
        The value of the series inductor in Henrys.
    C : Param
        The value of the shunt capacitor in Farads.
    """
    #: The value of the series inductor in Henrys.
    L: Param = param(constraint=Positive())
    
    #: The value of the shunt capacitor in Farads.
    C: Param = param(constraint=Positive())

    def a(self, freq: Frequency) -> jnp.ndarray:
        return jax.lax.cond(
            (self.L == 0.0) & (self.C == 0.0),
            lambda: self.a_thru(freq),
            lambda: self.a_general(freq),
        )

    def a_general(self, freq: Frequency) -> jnp.ndarray:
        """
        Internal calculation for the L-section ABCD matrix.
        """
        w = freq.w
        Z = 1j * w * self.L
        Y = 1j * w * self.C
        ones = jnp.ones_like(Z)

        # ABCD of Series Z cascaded with Shunt Y
        return jnp.array([
            [1 + Z * Y, Z],
            [Y,         ones]
        ]).transpose(2, 0, 1)

    def a_thru(self, freq: Frequency) -> jnp.ndarray:
        """
        Internal calculation for the case where both L and C are zero (ideal thru).
        """
        w = freq.w
        
        ones = jnp.ones_like(w, dtype=complex)
        zeros = jnp.zeros_like(w, dtype=complex)
        
        return jnp.array([
            [ones,  zeros],
            [zeros, ones]
        ]).transpose(2, 0, 1)

    def s(self, freq: Frequency) -> jnp.ndarray:
        from pmrf.rf import a2s
        return a2s(self.a(freq), self.z0)