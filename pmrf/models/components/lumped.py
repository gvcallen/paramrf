"""
Lumped elements (resistors, capacitors, inductors).
"""

import jax.numpy as jnp

from pmrf.parameters import Parameter
from pmrf.models.model import Model
from pmrf.frequency import Frequency

class Load(Model):
    """
    An abstract base class for N-port loads defined by their reflection coefficient.
    """
    gamma: float
    nports: int = 1
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        gamma, nports = self.gamma, self.nports
        # Create a frequency-dependent 1x1 matrix from the scalar gamma
        s = jnp.array(gamma).reshape(-1, 1, 1) * \
            jnp.eye(nports, dtype=jnp.complex128).reshape((-1, nports, nports)).\
            repeat(freq.npoints, 0)
        return s
 
class Capacitor(Model):
    """
    A 2-port model of a series capacitor.
    """
    C: Parameter = 1.0

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C

        z0_0 = z0_1 = self.z0
        denom = 1.0 + 1j * w * C * (z0_0 + z0_1)
        s11 = (1.0 - 1j * w * C * (jnp.conj(z0_0) - z0_1) ) / denom
        s22 = (1.0 - 1j * w * C * (jnp.conj(z0_1) - z0_0) ) / denom
        s12 = s21 = (2j * w * C * (z0_0.real * z0_1.real)**0.5) / denom

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
              
class Inductor(Model):
    """
    A 2-port model of a series inductor.
    """
    L: Parameter = 1.0
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        L = self.L
        w = freq.w

        z0_0 = z0_1 = self.z0
        
        denom = (1j * w * L) + (z0_0 + z0_1)
        s11 = (1j * w * L - jnp.conj(z0_0) + z0_1) / denom
        s22 = (1j * w * L + z0_0 - jnp.conj(z0_1)) / denom
        s12 = s21 = 2 * (z0_0.real * z0_1.real)**0.5 / denom

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s   

class Resistor(Model):
    """
    **Overview**

    A 2-port model of a series resistor.
    """
    R: Parameter = 1.0
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        z0_0 = z0_1 = self.z0
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        denom = R + (z0_0 + z0_1)
        s11 = ((R - jnp.conj(z0_0) + z0_1) / denom) * ones
        s22 = ((R + z0_0 - jnp.conj(z0_1)) / denom) * ones
        s12 = (2 * (z0_0.real * z0_1.real)**0.5 / denom) * ones
        s21 = s12

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
    
class ShuntCapacitor(Model):
    """
    A 2-port model of a shunt capacitor.

    This model represents a capacitor connected from the signal path to the ground
    reference, placed between port 1 and port 2.
    """
    C: Parameter = 1.0

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C
        
        z0 = self.z0
        yn = 1j * w * C * z0
        denom = 2.0 + yn

        s11 = -yn / denom
        s22 = s11
        s21 = 2.0 / denom
        s12 = s21

        # Construct the S-parameter matrix
        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1) # Transpose to (num_freq_points, 2, 2)

        return s 

class ShuntResistor(Model):
    """
    A 2-port model of a shunt resistor.

    This model represents a resistor connected from the signal path to the ground
    reference, placed between port 1 and port 2.
    """
    nports = 2
    R: Parameter = 50.0

    def s(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        
        z0 = self.z0

        yn = z0 / R

        denom = 2.0 + yn

        s11 = -yn / denom
        s22 = s11
        s21 = 2.0 / denom
        s12 = s21

        s11_freq = jnp.full(freq.w.shape, s11)
        s21_freq = jnp.full(freq.w.shape, s21)

        s = jnp.array([
            [s11_freq, s21_freq],
            [s21_freq, s11_freq]
        ]).transpose(2, 0, 1)

        return s

class ShuntInductor(Model):
    """
    A 2-port model of a shunt inductor.
    """
    nports = 2
    L: Parameter = 1e-9

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        L = self.L
        z0 = self.z0

        yn = z0 / (1j * w * L)
        denom = 2.0 + yn

        s11 = -yn / denom
        s22 = s11
        s21 = 2.0 / denom
        s12 = s21

        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
    
SHORT = Load(-1.0)
OPEN = Load(1.0)
MATCH = Load(0.0)