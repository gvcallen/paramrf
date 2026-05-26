"""
Lumped elements (resistors, capacitors, inductors).
"""

import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.types import ArrayLike, Param
from pmrf.parameters import param

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
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        R = self.R
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in = z0[..., 0]
            z_out = z0[..., 1]

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

    def y(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        Y = 1.0 / R
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        y11 = Y * ones
        y22 = Y * ones
        y12 = -Y * ones
        y21 = -Y * ones
        
        y = jnp.array([
            [y11, y12],
            [y21, y22]
        ]).transpose(2, 0, 1)
        
        return y
    

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

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        R = self.R
        Y = 1.0 / R
        
        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in, z_out = z0[..., 0], z0[..., 1]

        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        denom = z_in + z_out + Y * z_in * z_out
        
        s11 = ((z_out - jnp.conj(z_in) - Y * jnp.conj(z_in) * z_out) / denom) * ones
        s22 = ((z_in - jnp.conj(z_out) - Y * z_in * jnp.conj(z_out)) / denom) * ones
        s21 = ((2.0 * (z_in.real * z_out.real)**0.5) / denom) * ones
        s12 = s21

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
    

class Impedance(Model):
    """
    A 2-port model of a generic series impedance (R + jX).

    Parameters
    ----------
    R : Param
        The resistance in Ohms.
    X : Param
        The reactance in Ohms.
    """
    #: Resistance in Ohms
    R: Param = param()
    
    #: Reactance in Ohms
    X: Param = param()
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        Z = self.R + 1j * self.X
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in = z0[..., 0]
            z_out = z0[..., 1]

        denom_c = Z + (z_in + z_out)
        s_c11 = ((Z - jnp.conj(z_in) + z_out) / denom_c) * ones
        s_c22 = ((Z + z_in - jnp.conj(z_out)) / denom_c) * ones
        s_c12 = (2 * (z_in.real * z_out.real)**0.5 / denom_c) * ones
        s_c21 = s_c12

        s = jnp.array([
            [s_c11, s_c12],
            [s_c21, s_c22]
        ]).transpose(2, 0, 1)

        return s
    
    def y(self, freq: Frequency) -> jnp.ndarray:
        Z = self.R + 1j * self.X
        
        # Safely handle a 0-Ohm series short circuit (Y approaches infinity)
        Y = jnp.where(jnp.abs(Z) == 0.0, jnp.inf + 0j, 1.0 / Z)
        
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        y_val = Y * ones
        
        y = jnp.array([
            [y_val, -y_val],
            [-y_val, y_val]
        ]).transpose(2, 0, 1)
        
        return y
    

class Admittance(Model):
    """
    A 2-port model of a generic series admittance (G + jB).

    Parameters
    ----------
    G : Param
        The conductance in Siemens.
    B : Param
        The susceptance in Siemens.
    """
    #: Conductance in Siemens
    G: Param = param()
    
    #: Susceptance in Siemens
    B: Param = param()
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        Y = self.G + 1j * self.B
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in = z0[..., 0]
            z_out = z0[..., 1]

        denom_c = 1.0 + Y * (z_in + z_out)
        s_c11 = ((1.0 - Y * jnp.conj(z_in) + Y * z_out) / denom_c) * ones
        s_c22 = ((1.0 + Y * z_in - Y * jnp.conj(z_out)) / denom_c) * ones
        s_c12 = ((2.0 * (z_in.real * z_out.real)**0.5 * Y) / denom_c) * ones
        s_c21 = s_c12

        s = jnp.array([
            [s_c11, s_c12],
            [s_c21, s_c22]
        ]).transpose(2, 0, 1)

        return s    

    def y(self, freq: Frequency) -> jnp.ndarray:
        Y = self.G + 1j * self.B
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        y11 = Y * ones
        y22 = Y * ones
        y12 = -Y * ones
        y21 = -Y * ones
        
        y = jnp.array([
            [y11, y12],
            [y21, y22]
        ]).transpose(2, 0, 1)
        
        return y

 
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

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        w = freq.w
        C = self.C

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in = z0[..., 0]
            z_out = z0[..., 1]
        
        denom_c = 1.0 + 1j * w * C * (z_in + z_out)
        s_c11 = (1.0 - 1j * w * C * (jnp.conj(z_in) - z_out) ) / denom_c
        s_c22 = (1.0 - 1j * w * C * (jnp.conj(z_out) - z_in) ) / denom_c
        s_c12 = s_c21 = (2j * w * C * (z_in.real * z_out.real)**0.5) / denom_c

        s = jnp.array([
            [s_c11, s_c12],
            [s_c21, s_c22]
        ]).transpose(2, 0, 1)

        return s

    def y(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C
        Y = 1j * w * C
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        y11 = Y * ones
        y22 = Y * ones
        y12 = -Y * ones
        y21 = -Y * ones
        
        y = jnp.array([
            [y11, y12],
            [y21, y22]
        ]).transpose(2, 0, 1)
        
        return y
    

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

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        w = freq.w
        C = self.C
        Q = self.Q

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in = z0[..., 0]
            z_out = z0[..., 1]
        
        Z_scaled = 1.0 + 1j * (1.0 / Q)
        
        denom = Z_scaled + 1j * w * C * (z_in + z_out)
        
        s11 = (Z_scaled + 1j * w * C * (-jnp.conj(z_in) + z_out)) / denom
        s22 = (Z_scaled + 1j * w * C * (z_in - jnp.conj(z_out))) / denom
        s12 = s21 = (1j * w * C * 2.0 * (z_in.real * z_out.real)**0.5) / denom

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
    
    def y(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C
        Q = self.Q
        
        Z_scaled = 1.0 + 1j * (1.0 / Q)
        Z_component = Z_scaled / (1j * w * C)
        Y = jnp.where(w == 0, 0.0 + 0j, 1.0 / Z_component)
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        y11 = Y * ones
        y22 = Y * ones
        y12 = -Y * ones
        y21 = -Y * ones
        
        y = jnp.array([
            [y11, y12],
            [y21, y22]
        ]).transpose(2, 0, 1)
        
        return y


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

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        w = freq.w
        C = self.C
        Y = 1j * w * C
        
        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in, z_out = z0[..., 0], z0[..., 1]
        
        denom = z_in + z_out + Y * z_in * z_out
        
        s11 = (z_out - jnp.conj(z_in) - Y * jnp.conj(z_in) * z_out) / denom
        s22 = (z_in - jnp.conj(z_out) - Y * z_in * jnp.conj(z_out)) / denom
        s21 = (2.0 * (z_in.real * z_out.real)**0.5) / denom
        s12 = s21

        s = jnp.array([
            [s11, s12],
            [s21, s22]
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
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        L = self.L
        w = freq.w

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in = z0[..., 0]
            z_out = z0[..., 1]

        denom_c = (1j * w * L) + (z_in + z_out)
        s_c11 = (1j * w * L - jnp.conj(z_in) + z_out) / denom_c
        s_c22 = (1j * w * L + z_in - jnp.conj(z_out)) / denom_c
        s_c12 = s_c21 = 2 * (z_in.real * z_out.real)**0.5 / denom_c

        s = jnp.array([
            [s_c11, s_c12],
            [s_c21, s_c22]
        ]).transpose(2, 0, 1)

        return s
    
    def y(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        L = self.L
        
        Y = jnp.where(w == 0, jnp.inf + 0j, 1.0 / (1j * w * L))
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        y_val = Y * ones
        
        y = jnp.array([
            [y_val, -y_val],
            [-y_val, y_val]
        ]).transpose(2, 0, 1)
        
        return y

    
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

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        w = freq.w
        L = self.L
        Q = self.Q

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in = z0[..., 0]
            z_out = z0[..., 1]
        
        # Total impedance Z = w*L/Q + j*w*L
        Z = w * L * (1.0 / Q + 1j)

        denom = Z + (z_in + z_out)
        
        s11 = (Z - jnp.conj(z_in) + z_out) / denom
        s22 = (Z + z_in - jnp.conj(z_out)) / denom
        s12 = s21 = (2.0 * (z_in.real * z_out.real)**0.5) / denom

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
    
    def y(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        L = self.L
        Q = self.Q
        
        Z = w * L * (1.0 / Q + 1j)
        Y = jnp.where(w == 0, jnp.inf + 0j, 1.0 / Z)
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        y_val = Y * ones
        
        y = jnp.array([
            [y_val, -y_val],
            [-y_val, y_val]
        ]).transpose(2, 0, 1)
        
        return y


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

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        w = freq.w
        L = self.L
        Z = 1j * w * L

        if jnp.isscalar(z0):
            z_in = z_out = z0
        else:
            z_in, z_out = z0[..., 0], z0[..., 1]

        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
        
        denom = Z * (z_in + z_out) + z_in * z_out
        
        s11 = ((Z * (z_out - jnp.conj(z_in)) - jnp.conj(z_in) * z_out) / denom) * ones
        s22 = ((Z * (z_in - jnp.conj(z_out)) - z_in * jnp.conj(z_out)) / denom) * ones
        s21 = ((Z * 2.0 * (z_in.real * z_out.real)**0.5) / denom) * ones
        s12 = s21

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s            