"""
Lumped elements (resistors, capacitors, inductors).
"""

import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.parameters import Param, param

class Load(Model):
    """
    An class for N-port loads defined by their reflection coefficient.
    """
    #: The reflection coefficient (e.g., 0.0 for match, 1.0 for open, -1.0 for short).
    gamma: Param = param()
    #: The number of ports this load presents. Default is 1.
    nports: int = 1
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        gamma, nports = self.gamma, self.nports
        # Create a frequency-dependent 1x1 matrix from the scalar gamma
        s = jnp.array(gamma).reshape(-1, 1, 1) * \
            jnp.eye(nports, dtype=jnp.complex128).reshape((-1, nports, nports)).\
            repeat(freq.npoints, 0)
        return s
    

class FixedLoad(Model):
    """
    An class for N-port loads defined by fixed (non-tunable) reflection coefficient.
    """
    #: The complex reflection coefficient (e.g., 0.0 for match, 1.0 for open, -1.0 for short).
    gamma: float
    #: The number of ports this load presents. Default is 1.
    nports: int = 1
    
    def __call__(self) -> jnp.ndarray:
        return Load(gamma=self.gamma, nports=self.nports)


class Resistor(Model):
    """
    A 2-port model of a series resistor.
    """
    #: The resistance in Ohms.
    R: Param = param(positive=True)
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        # Parse reference impedances safely
        if jnp.isscalar(self.z0):
            z_in = z_out = self.z0
        else:
            z_in = self.z0[..., 0]
            z_out = self.z0[..., 1]

        # Component path
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
    """
    #: The capacitance in Farads.
    C: Param = param(positive=True)

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C

        if jnp.isscalar(self.z0):
            z_in = z_out = self.z0
        else:
            z_in = self.z0[..., 0]
            z_out = self.z0[..., 1]
        
        # Component path
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
    """
    #: The inductance in Henrys.
    L: Param = param(positive=True)
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        L = self.L
        w = freq.w

        if jnp.isscalar(self.z0):
            z_in = z_out = self.z0
        else:
            z_in = self.z0[..., 0]
            z_out = self.z0[..., 1]

        # Component path
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
    """
    #: The resistance in Ohms.
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
    """
    #: The capacitance in Farads
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
    """
    #: The inductance in Henrys
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
    """
    #: The inductance in Henrys
    L: Param = param()
    #: The quality factor representing non-ideal losses
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
    """
    #: The capacitance in Farads
    C: Param = param()
    #: The quality factor representing non-ideal losses. Default is 50.0.
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
    
def Short(nports=1):
    """A standard ideal short circuit load (gamma = -1.0)."""
    return FixedLoad(-1.0, nports=nports)

def Open(nports=1):
    """A standard ideal open circuit load (gamma = 1.0)."""
    return FixedLoad(1.0, nports=nports)

def Match(nports=1):
    """A standard ideal open circuit load (gamma = 1.0)."""
    return FixedLoad(0.0, nports=nports)