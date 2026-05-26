"""
Specific sections layouts such as Pi or T, and some common specializations.
"""
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.types import Param
from pmrf.parameters import param

class PiSection(Model):
    """
    A 2-port model of a general Pi-network.

    Defined entirely by branch admittances, matching standard network theory conventions.

    Parameters
    ----------
    Y1 : Param
        Admittance of the first shunt branch.
    Y2 : Param
        Admittance of the second shunt branch.
    Y3 : Param
        Admittance of the series branch connecting the two shunt branches.
    """
    #: Admittance of the first shunt branch.
    Y1: Param = param()
    
    #: Admittance of the second shunt branch.
    Y2: Param = param()
    
    #: Admittance of the series branch.
    Y3: Param = param()

    def y(self, freq: Frequency) -> jnp.ndarray:
        # The native domain of a Pi-network
        ones = jnp.ones(freq.npoints, dtype=complex)
        Y1 = self.Y1 * ones
        Y2 = self.Y2 * ones
        Y3 = self.Y3 * ones

        return jnp.array([
            [Y1 + Y3,  -Y3],
            [-Y3,      Y2 + Y3],
        ]).transpose(2, 0, 1)


class BoxSection(Model):
    """
    A 4-port model of a general Box-network.

    Parameters
    ----------
    Y1 : Param
        Admittance of the first shunt branch (Port 1 to Port 3).
    Y2 : Param
        Admittance of the second shunt branch (Port 2 to Port 4).
    Y3 : Param
        Admittance of the series branch (Port 1 to Port 2).
    Y4 : Param
        Admittance of the bridging branch (Port 3 to Port 4).
    """
    #: Admittance of the first shunt branch.
    Y1: Param = param()
    
    #: Admittance of the second shunt branch.
    Y2: Param = param()
    
    #: Admittance of the series branch.
    Y3: Param = param()
    
    #: Admittance of the bridging branch.
    Y4: Param = param()

    def y(self, freq: Frequency) -> jnp.ndarray:
        # Broadcast scalar params to frequency arrays
        ones = jnp.ones(freq.npoints, dtype=complex)
        Y1 = self.Y1 * ones
        Y2 = self.Y2 * ones
        Y3 = self.Y3 * ones
        Y4 = self.Y4 * ones
        
        zero = jnp.zeros_like(Y1)

        # Because Box uses native admittances for a y-matrix, no divisions are required.
        return jnp.array([
            [Y1 + Y3,   -Y3,        -Y1,        zero],
            [-Y3,       Y2 + Y3,    zero,       -Y2],
            [-Y1,       zero,       Y1 + Y4,    -Y4],
            [zero,      -Y2,        -Y4,        Y2 + Y4]
        ]).transpose(2, 0, 1)


class TSection(Model):
    """
    A 2-port model of a general Tee-network.

    Defined entirely by branch impedances, matching standard network theory conventions.

    Parameters
    ----------
    Z1 : Param
        Impedance of the first series branch.
    Z2 : Param
        Impedance of the second series branch.
    Z3 : Param
        Impedance of the shunt branch to ground.
    """
    #: Impedance of the first series branch.
    Z1: Param = param()
    
    #: Impedance of the second series branch.
    Z2: Param = param()
    
    #: Impedance of the shunt branch.
    Z3: Param = param()

    def z(self, freq: Frequency) -> jnp.ndarray:
        # The native domain of a T-network
        ones = jnp.ones(freq.npoints, dtype=complex)
        Z1 = self.Z1 * ones
        Z2 = self.Z2 * ones
        Z3 = self.Z3 * ones

        return jnp.array([
            [Z1 + Z3, Z3],
            [Z3,      Z2 + Z3],
        ]).transpose(2, 0, 1)


class LSection(Model):
    """
    A 2-port model of a general L-section impedance matching network.
    
    Uses a series impedance followed by a shunt admittance.

    Parameters
    ----------
    Z : Param
        Impedance of the series branch.
    Y : Param
        Admittance of the shunt branch.
    """
    #: Impedance of the series branch.
    Z: Param = param()
    
    #: Admittance of the shunt branch.
    Y: Param = param()

    def a(self, freq: Frequency) -> jnp.ndarray:
        # Broadcast scalar params
        ones = jnp.ones(freq.npoints, dtype=complex)
        Z = self.Z * ones
        Y = self.Y * ones

        # Perfectly stable without divisions
        A = 1 + Z * Y
        B = Z
        C = Y
        D = ones

        return jnp.array([
            [A, B],
            [C, D],
        ]).transpose(2, 0, 1)


class PiSectionCLC(Model):
    """
    A 2-port model of a Pi-network with a Capacitor-Inductor-Capacitor topology.
    """
    C1: Param = param()
    L: Param = param()
    C2: Param = param()

    def a(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        w2 = w**2
        C1, C2, L = self.C1, self.C2, self.L

        A = 1 - w2 * C2 * L
        B = 1j * w * L
        C = 1j * w * (C1 + C2 - w2 * C1 * C2 * L)
        D = 1 - w2 * C1 * L

        return jnp.array([
            [A, B],
            [C, D],
        ]).transpose(2, 0, 1)


class BoxSectionCLCC(Model):
    """
    A 4-port model of a Box-network with a Capacitor-Inductor-Capacitor-Capacitor topology.
    """    
    C1: Param = param()
    L: Param = param()
    C2: Param = param()
    C3: Param = param()

    def y(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        Y1 = 1j * w * self.C1
        Y2 = 1j * w * self.C2
        Y4 = 1j * w * self.C3
        
        # We must divide by wL here to get admittance. 
        # Safely generating inf prevents NaN propagation in JAX linear solvers if DC is hit.
        Y3 = jnp.where(w * self.L == 0.0, jnp.inf + 0j, 1.0 / (1j * w * self.L))
        
        zero = jnp.zeros_like(Y1)

        return jnp.array([
            [Y1 + Y3,       -Y3,            -Y1,            zero],
            [-Y3,           Y2 + Y3,        zero,           -Y2],
            [-Y1,           zero,           Y1 + Y4,        -Y4],
            [zero,          -Y2,            -Y4,            Y2 + Y4]
        ]).transpose(2, 0, 1)
    

class TSectionLCL(Model):
    """
    A 2-port model of a Tee-network with an Inductor-Capacitor-Inductor topology.
    """
    L1: Param = param()
    C: Param = param()
    L2: Param = param()

    def a(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        w2 = w**2
        L1, C, L2 = self.L1, self.C, self.L2

        A = 1 - w2 * L1 * C
        B = 1j * w * (L1 + L2 - w2 * L1 * L2 * C)
        C_term = 1j * w * C
        D = 1 - w2 * L2 * C

        return jnp.array([
            [A, B],
            [C_term, D],
        ]).transpose(2, 0, 1)


class LSectionLC(Model):
    """
    A 2-port model of an L-section impedance matching network.
    """
    L: Param = param()
    C: Param = param()

    def a(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        w2 = w**2
        L, C = self.L, self.C

        A = 1 - w2 * L * C
        B = 1j * w * L
        C_term = 1j * w * C
        D = jnp.ones_like(w, dtype=complex)

        return jnp.array([
            [A, B],
            [C_term, D],
        ]).transpose(2, 0, 1)