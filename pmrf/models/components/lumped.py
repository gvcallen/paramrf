"""
Lumped elements (resistors, capacitors, inductors).
"""

import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.constraints import Positive
from pmrf.parameters import Param, param
from pmrf.utils import field

class Load(Model):
    """
    A class for N-port loads defined by their reflection coefficient.
    
    Parameters
    ----------
    gamma : Param
        The reflection coefficient (e.g., 0.0 for match, 1.0 for open, -1.0 for short).
    nports : int
        The number of ports this load presents. Default is 1.
    """
    #: The reflection coefficient
    gamma: Param = param()
    
    #: Number of ports
    nports: int = 1
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        gamma, nports = self.gamma, self.nports
        s = jnp.array(gamma).reshape(-1, 1, 1) * \
            jnp.eye(nports, dtype=jnp.complex128).reshape((-1, nports, nports)).\
            repeat(freq.npoints, 0)
        return s
    

class ConstantLoad(Model):
    """
    A class for N-port loads defined by constant (non-tunable) reflection coefficient.

    This bakes in the "non-tunability" into the load by making it a float
    instead of a parameter.
    
    Parameters
    ----------
    gamma : float
        The complex reflection coefficient (e.g., 0.0 for match, 1.0 for open, -1.0 for short).
    nports : int
        The number of ports this load presents. Default is 1.
    """
    #: Complex reflection coefficient
    gamma: float = field(static=True)
    
    #: Number of ports
    nports: int = 1
    
    def build(self) -> jnp.ndarray:
        return Load(gamma=self.gamma, nports=self.nports)


class Resistor(Model):
    """
    A 2-port model of a series resistor.

    Parameters
    ----------
    R : Param
        The resistance in Ohms.
    """
    #: Resistance in Ohms
    R: Param = param()
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        if jnp.isscalar(self.z0):
            z_in = z_out = self.z0
        else:
            z_in = self.z0[..., 0]
            z_out = self.z0[..., 1]

        denom_c = R + (z_in + z_out)
        s_c11 = ((R - jnp.conj(z_in) + z_out) / denom_c) * ones
        s_c22 = ((R + z_in - jnp.conj(z_out)) / denom_c) * ones
        s_c12 = (2 * (z_in.real * z_out.real)**0.5 / denom_c) * ones
        s_c21 = s_c12

        s = jnp.array([
            [s_c11, s_c12],
            [s_c21, s_c22]
        ]).transpose(2, 0, 1)

        return s    
 
 
class Capacitor(Model):
    """
    A 2-port model of a series capacitor.

    Parameters
    ----------
    C : Param
        The capacitance in Farads.
    """
    #: Capacitance in Farads
    C: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C

        if jnp.isscalar(self.z0):
            z_in = z_out = self.z0
        else:
            z_in = self.z0[..., 0]
            z_out = self.z0[..., 1]
        
        denom_c = 1.0 + 1j * w * C * (z_in + z_out)
        s_c11 = (1.0 - 1j * w * C * (jnp.conj(z_in) - z_out) ) / denom_c
        s_c22 = (1.0 - 1j * w * C * (jnp.conj(z_out) - z_in) ) / denom_c
        s_c12 = s_c21 = (2j * w * C * (z_in.real * z_out.real)**0.5) / denom_c

        s = jnp.array([
            [s_c11, s_c12],
            [s_c21, s_c22]
        ]).transpose(2, 0, 1)

        return s
            
              
class Inductor(Model):
    """
    A 2-port model of a series inductor.

    Parameters
    ----------
    L : Param
        The inductance in Henrys.
    """
    #: Inductance in Henrys
    L: Param = param()
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        L = self.L
        w = freq.w

        if jnp.isscalar(self.z0):
            z_in = z_out = self.z0
        else:
            z_in = self.z0[..., 0]
            z_out = self.z0[..., 1]

        denom_c = (1j * w * L) + (z_in + z_out)
        s_c11 = (1j * w * L - jnp.conj(z_in) + z_out) / denom_c
        s_c22 = (1j * w * L + z_in - jnp.conj(z_out)) / denom_c
        s_c12 = s_c21 = 2 * (z_in.real * z_out.real)**0.5 / denom_c

        s = jnp.array([
            [s_c11, s_c12],
            [s_c21, s_c22]
        ]).transpose(2, 0, 1)

        return s
    

class ShuntResistor(Model):
    """
    A 2-port model of a shunt resistor shunting to ground.

    Parameters
    ----------
    R : Param
        The resistance in Ohms.
    """
    #: Resistance in Ohms
    R: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        Y = 1.0 / R
        
        if jnp.isscalar(self.z0):
            z0_0 = z0_1 = self.z0
        else:
            z0_0, z0_1 = self.z0[..., 0], self.z0[..., 1]

        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        denom = z0_0 + z0_1 + Y * z0_0 * z0_1
        
        s11 = ((z0_1 - jnp.conj(z0_0) - Y * jnp.conj(z0_0) * z0_1) / denom) * ones
        s22 = ((z0_0 - jnp.conj(z0_1) - Y * z0_0 * jnp.conj(z0_1)) / denom) * ones
        s21 = ((2.0 * (z0_0.real * z0_1.real)**0.5) / denom) * ones
        s12 = s21

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s    
    
    
class ShuntCapacitor(Model):
    """
    A 2-port model of a shunt capacitor shunting to ground.

    Parameters
    ----------
    C : Param
        The capacitance in Farads
    """
    #: Capacitance in Farads
    C: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C
        Y = 1j * w * C
        
        if jnp.isscalar(self.z0):
            z0_0 = z0_1 = self.z0
        else:
            z0_0, z0_1 = self.z0[..., 0], self.z0[..., 1]
        
        denom = z0_0 + z0_1 + Y * z0_0 * z0_1
        
        s11 = (z0_1 - jnp.conj(z0_0) - Y * jnp.conj(z0_0) * z0_1) / denom
        s22 = (z0_0 - jnp.conj(z0_1) - Y * z0_0 * jnp.conj(z0_1)) / denom
        s21 = (2.0 * (z0_0.real * z0_1.real)**0.5) / denom
        s12 = s21

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s                


class ShuntInductor(Model):
    """
    A 2-port model of a shunt inductor shunting to ground. 
    Internally uses Z-formulation to prevent divide-by-zero errors at L=0 or DC.

    Parameters
    ----------
    L : Param
        The inductance in Henrys
    """
    #: Inductance in Henrys
    L: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        L = self.L
        Z = 1j * w * L

        if jnp.isscalar(self.z0):
            z0_0 = z0_1 = self.z0
        else:
            z0_0, z0_1 = self.z0[..., 0], self.z0[..., 1]

        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        denom = Z * (z0_0 + z0_1) + z0_0 * z0_1
        
        s11 = ((Z * (z0_1 - jnp.conj(z0_0)) - jnp.conj(z0_0) * z0_1) / denom) * ones
        s22 = ((Z * (z0_0 - jnp.conj(z0_1)) - z0_0 * jnp.conj(z0_1)) / denom) * ones
        s21 = ((Z * 2.0 * (z0_0.real * z0_1.real)**0.5) / denom) * ones
        s12 = s21

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s            

    
class InductorQ(Model):
    """
    A 2-port model of a series inductor with a finite Quality Factor (Q).

    Parameters
    ----------
    L : Param
        The inductance in Henrys
    Q : Param
        The quality factor representing non-ideal losses
    """
    #: Inductance in Henrys
    L: Param = param()

    #: Quality factor
    Q: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        L = self.L
        Q = self.Q

        z0_0 = z0_1 = self.z0
        
        # Total impedance Z = w*L/Q + j*w*L
        Z = w * L * (1.0 / Q + 1j)

        denom = Z + (z0_0 + z0_1)
        
        s11 = (Z - jnp.conj(z0_0) + z0_1) / denom
        s22 = (Z + z0_0 - jnp.conj(z0_1)) / denom
        s12 = s21 = (2.0 * (z0_0.real * z0_1.real)**0.5) / denom

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s


class CapacitorQ(Model):
    """
    A 2-port model of a series capacitor with a finite Quality Factor (Q).

    Parameters
    ----------
    C : Param
        The capacitance in Farads
    Q : Param
        The quality factor representing non-ideal losses. Default is 50.0.
    """
    #: Capacitance in Farads
    C: Param = param()
    
    #: Quality factor
    Q: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C
        Q = self.Q

        z0_0 = z0_1 = self.z0
        
        Z_scaled = 1.0 + 1j * (1.0 / Q)
        
        denom = Z_scaled + 1j * w * C * (z0_0 + z0_1)
        
        s11 = (Z_scaled + 1j * w * C * (-jnp.conj(z0_0) + z0_1)) / denom
        s22 = (Z_scaled + 1j * w * C * (z0_0 - jnp.conj(z0_1))) / denom
        s12 = s21 = (1j * w * C * 2.0 * (z0_0.real * z0_1.real)**0.5) / denom

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
    
def Short(nports=1) -> ConstantLoad:
    """
    A standard ideal short circuit load (gamma = -1.0).
    
    Parameters
    ----------
    nports : int, default=1
        The number of ports for the load.    
    """
    return ConstantLoad(-1.0, nports=nports)

def Open(nports=1) -> ConstantLoad:
    """
    A standard ideal open circuit load (gamma = 1.0).
    
    Parameters
    ----------
    nports : int, default=1
        The number of ports for the load.
    """
    return ConstantLoad(1.0, nports=nports)

def Match(nports=1) -> ConstantLoad:
    """
    A standard ideal matched circuit load (gamma = 0.0).
    
    Parameters
    ----------
    nports : int, default=1
        The number of ports for the load.
    """
    return ConstantLoad(0.0, nports=nports)