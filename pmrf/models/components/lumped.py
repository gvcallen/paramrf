"""
Lumped elements (resistors, capacitors, inductors).
"""

import jax.numpy as jnp

from pmrf.parameters import Parameter, Fixed
from pmrf.model import Model
from pmrf.frequency import Frequency


class Load(Model):
    """
    An abstract base class for N-port loads defined by their reflection coefficient.

    Parameters:
    -----------
    gamma : Parameter
        The complex reflection coefficient (e.g., 0.0 for match, 1.0 for open, -1.0 for short).
    nports : int
        The number of ports this load presents. Default is 1.
    """
    gamma: Parameter
    nports: int = 1
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        gamma, nports = self.gamma, self.nports
        # Create a frequency-dependent 1x1 matrix from the scalar gamma
        s = jnp.array(gamma).reshape(-1, 1, 1) * \
            jnp.eye(nports, dtype=jnp.complex128).reshape((-1, nports, nports)).\
            repeat(freq.npoints, 0)
        return s
    
    
class Resistor(Model):
    """
    A 2-port or 4-port floating model of a series resistor.

    Parameters:
    -----------
    R : Parameter
        The resistance in Ohms. Default is 50.0.
    floating : bool
        If True, returns a 4-port network where the element is between 
        Port 0 and Port 2, and Port 1 is shorted to Port 3.
    """
    R: Parameter = 50.0
    floating: bool = False
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        # Parse reference impedances safely
        if jnp.isscalar(self.z0):
            z_0 = z_1 = z_2 = z_3 = self.z0
        else:
            z_0 = self.z0[..., 0]
            if self.floating and self.z0.shape[-1] >= 4:
                z_1, z_2, z_3 = self.z0[..., 1], self.z0[..., 2], self.z0[..., 3]
            else:
                z_1 = self.z0[..., 1]
                z_2 = z_3 = z_1  # Fallback

        z_in = z_0
        z_out = z_2 if self.floating else z_1

        # Component path (Port 0 to Port 2)
        denom_c = R + (z_in + z_out)
        s_c11 = ((R - jnp.conj(z_in) + z_out) / denom_c) * ones
        s_c22 = ((R + z_in - jnp.conj(z_out)) / denom_c) * ones
        s_c12 = (2 * (z_in.real * z_out.real)**0.5 / denom_c) * ones
        s_c21 = s_c12

        if not self.floating:
            s = jnp.array([
                [s_c11, s_c12],
                [s_c21, s_c22]
            ]).transpose(2, 0, 1)
        else:
            # Ideal through return path (Port 1 to Port 3)
            denom_t = z_1 + z_3
            s_t11 = ((z_3 - jnp.conj(z_1)) / denom_t) * ones
            s_t33 = ((z_1 - jnp.conj(z_3)) / denom_t) * ones
            s_t13 = (2 * (z_1.real * z_3.real)**0.5 / denom_t) * ones
            s_t31 = s_t13

            zeros = jnp.zeros_like(s_c11)

            s = jnp.array([
                [s_c11, zeros, s_c12, zeros],
                [zeros, s_t11, zeros, s_t13],
                [s_c21, zeros, s_c22, zeros],
                [zeros, s_t31, zeros, s_t33]
            ]).transpose(2, 0, 1)

        return s    
 
 
class Capacitor(Model):
    """
    A 2-port or 4-port floating model of a series capacitor.

    Parameters:
    -----------
    C : Parameter
        The capacitance in Farads. Default is 1.0e-12 (1 pF).
    floating : bool
    """
    C: Parameter = 1.0e-12
    floating: bool = False

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        if jnp.isscalar(self.z0):
            z_0 = z_1 = z_2 = z_3 = self.z0
        else:
            z_0 = self.z0[..., 0]
            if self.floating and self.z0.shape[-1] >= 4:
                z_1, z_2, z_3 = self.z0[..., 1], self.z0[..., 2], self.z0[..., 3]
            else:
                z_1 = self.z0[..., 1]
                z_2 = z_3 = z_1

        z_in = z_0
        z_out = z_2 if self.floating else z_1
        
        # Component path (Port 0 to Port 2)
        denom_c = 1.0 + 1j * w * C * (z_in + z_out)
        s_c11 = (1.0 - 1j * w * C * (jnp.conj(z_in) - z_out) ) / denom_c
        s_c22 = (1.0 - 1j * w * C * (jnp.conj(z_out) - z_in) ) / denom_c
        s_c12 = s_c21 = (2j * w * C * (z_in.real * z_out.real)**0.5) / denom_c

        if not self.floating:
            s = jnp.array([
                [s_c11, s_c12],
                [s_c21, s_c22]
            ]).transpose(2, 0, 1)
        else:
            # Ideal through return path (Port 1 to Port 3)
            denom_t = z_1 + z_3
            s_t11 = ((z_3 - jnp.conj(z_1)) / denom_t) * ones
            s_t33 = ((z_1 - jnp.conj(z_3)) / denom_t) * ones
            s_t13 = s_t31 = (2 * (z_1.real * z_3.real)**0.5 / denom_t) * ones

            zeros = jnp.zeros_like(s_c11)

            s = jnp.array([
                [s_c11, zeros, s_c12, zeros],
                [zeros, s_t11, zeros, s_t13],
                [s_c21, zeros, s_c22, zeros],
                [zeros, s_t31, zeros, s_t33]
            ]).transpose(2, 0, 1)

        return s
            
              
class Inductor(Model):
    """
    A 2-port or 4-port floating model of a series inductor.

    Parameters:
    -----------
    L : Parameter
        The inductance in Henrys. Default is 1.0e-9 (1 nH).
    floating : bool
    """
    L: Parameter = 1.0e-9
    floating: bool = False
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        L = self.L
        w = freq.w
        ones = jnp.ones(freq.npoints, dtype=jnp.complex128)

        if jnp.isscalar(self.z0):
            z_0 = z_1 = z_2 = z_3 = self.z0
        else:
            z_0 = self.z0[..., 0]
            if self.floating and self.z0.shape[-1] >= 4:
                z_1, z_2, z_3 = self.z0[..., 1], self.z0[..., 2], self.z0[..., 3]
            else:
                z_1 = self.z0[..., 1]
                z_2 = z_3 = z_1
        
        z_in = z_0
        z_out = z_2 if self.floating else z_1

        # Component path (Port 0 to Port 2)
        denom_c = (1j * w * L) + (z_in + z_out)
        s_c11 = (1j * w * L - jnp.conj(z_in) + z_out) / denom_c
        s_c22 = (1j * w * L + z_in - jnp.conj(z_out)) / denom_c
        s_c12 = s_c21 = 2 * (z_in.real * z_out.real)**0.5 / denom_c

        if not self.floating:
            s = jnp.array([
                [s_c11, s_c12],
                [s_c21, s_c22]
            ]).transpose(2, 0, 1)
        else:
            # Ideal through return path (Port 1 to Port 3)
            denom_t = z_1 + z_3
            s_t11 = ((z_3 - jnp.conj(z_1)) / denom_t) * ones
            s_t33 = ((z_1 - jnp.conj(z_3)) / denom_t) * ones
            s_t13 = s_t31 = (2 * (z_1.real * z_3.real)**0.5 / denom_t) * ones

            zeros = jnp.zeros_like(s_c11)

            s = jnp.array([
                [s_c11, zeros, s_c12, zeros],
                [zeros, s_t11, zeros, s_t13],
                [s_c21, zeros, s_c22, zeros],
                [zeros, s_t31, zeros, s_t33]
            ]).transpose(2, 0, 1)

        return s
    

class ShuntResistor(Model):
    """
    A model of a shunt resistor.

    Parameters:
    -----------
    R : Parameter
        The resistance in Ohms. Default is 50.0.
    floating : bool
        If False (default), returns a standard 2-port model shunting to ground.
        If True, returns a 4-port differential model where the resistor bridges
        the top and bottom lines. Ports 0 & 1 act as the input pair, and Ports
        2 & 3 act as the output pair.
    """
    R: Parameter = 50.0
    floating: bool = False

    def s(self, freq: Frequency) -> jnp.ndarray:
        R = self.R
        Y = 1.0 / R
        
        if not self.floating:
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
        else:    
            ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
            
            # Handle generalized z0
            if jnp.isscalar(self.z0):
                z0_0 = z0_1 = z0_2 = z0_3 = self.z0
            else:
                z0_0, z0_1 = self.z0[..., 0], self.z0[..., 1]
                z0_2, z0_3 = self.z0[..., 2], self.z0[..., 3]

            # Port admittances
            Y0, Y1 = 1.0 / z0_0, 1.0 / z0_1
            Y2, Y3 = 1.0 / z0_2, 1.0 / z0_3
            
            # Port real resistances
            R0, R1 = z0_0.real, z0_1.real
            R2, R3 = z0_2.real, z0_3.real

            # Nodal admittances (Top line connects 0 and 2, Bottom connects 1 and 3)
            YA = Y0 + Y2
            YB = Y1 + Y3
            
            # Determinant of the nodal admittance matrix
            D = YA * YB + Y * (YA + YB)
            
            # Node voltage response factors
            V_AA = (YB + Y) / D
            V_BB = (YA + Y) / D
            V_AB = Y / D

            def K(r_i, r_j, y_i, y_j):
                return 2.0 * jnp.sqrt(r_i * r_j) * y_i * y_j

            # Base Reflections
            s00 = (K(R0, R0, Y0, Y0) * V_AA - (jnp.conj(z0_0) / z0_0)) * ones
            s11 = (K(R1, R1, Y1, Y1) * V_BB - (jnp.conj(z0_1) / z0_1)) * ones
            s22 = (K(R2, R2, Y2, Y2) * V_AA - (jnp.conj(z0_2) / z0_2)) * ones
            s33 = (K(R3, R3, Y3, Y3) * V_BB - (jnp.conj(z0_3) / z0_3)) * ones

            # Transmission along the same continuous line
            s20 = s02 = (K(R0, R2, Y0, Y2) * V_AA) * ones
            s31 = s13 = (K(R1, R3, Y1, Y3) * V_BB) * ones

            # Cross-coupling through the shunt resistor
            s10 = s01 = (K(R0, R1, Y0, Y1) * V_AB) * ones
            s30 = s03 = (K(R0, R3, Y0, Y3) * V_AB) * ones
            s21 = s12 = (K(R1, R2, Y1, Y2) * V_AB) * ones
            s32 = s23 = (K(R2, R3, Y2, Y3) * V_AB) * ones

            # Assemble the 4x4 matrix using standard Input/Output block mappings
            s = jnp.array([
                [s00, s01, s02, s03],
                [s10, s11, s12, s13],
                [s20, s21, s22, s23],
                [s30, s31, s32, s33]
            ]).transpose(2, 0, 1)

            return s    
    
    
class ShuntCapacitor(Model):
    """
    A model of a shunt capacitor.

    Parameters:
    -----------
    C : Parameter
        The capacitance in Farads. Default is 1.0e-12 (1 pF).
    floating : bool
        If False (default), returns a standard 2-port model shunting to ground.
        If True, returns a 4-port differential model where the capacitor bridges
        the top and bottom lines. Ports 0 & 1 act as the input pair, and Ports
        2 & 3 act as the output pair.
    """
    C: Parameter = 1.0e-12
    floating: bool = False

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C
        Y = 1j * w * C
        
        if not self.floating:
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
        else:
            ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
            
            if jnp.isscalar(self.z0):
                z0_0 = z0_1 = z0_2 = z0_3 = self.z0
            else:
                z0_0, z0_1 = self.z0[..., 0], self.z0[..., 1]
                z0_2, z0_3 = self.z0[..., 2], self.z0[..., 3]

            Y0, Y1 = 1.0 / z0_0, 1.0 / z0_1
            Y2, Y3 = 1.0 / z0_2, 1.0 / z0_3
            
            R0, R1 = z0_0.real, z0_1.real
            R2, R3 = z0_2.real, z0_3.real

            YA = Y0 + Y2
            YB = Y1 + Y3
            
            D = YA * YB + Y * (YA + YB)
            
            V_AA = (YB + Y) / D
            V_BB = (YA + Y) / D
            V_AB = Y / D

            def K(r_i, r_j, y_i, y_j):
                return 2.0 * jnp.sqrt(r_i * r_j) * y_i * y_j

            s00 = (K(R0, R0, Y0, Y0) * V_AA - (jnp.conj(z0_0) / z0_0)) * ones
            s11 = (K(R1, R1, Y1, Y1) * V_BB - (jnp.conj(z0_1) / z0_1)) * ones
            s22 = (K(R2, R2, Y2, Y2) * V_AA - (jnp.conj(z0_2) / z0_2)) * ones
            s33 = (K(R3, R3, Y3, Y3) * V_BB - (jnp.conj(z0_3) / z0_3)) * ones

            s20 = s02 = (K(R0, R2, Y0, Y2) * V_AA) * ones
            s31 = s13 = (K(R1, R3, Y1, Y3) * V_BB) * ones

            s10 = s01 = (K(R0, R1, Y0, Y1) * V_AB) * ones
            s30 = s03 = (K(R0, R3, Y0, Y3) * V_AB) * ones
            s21 = s12 = (K(R1, R2, Y1, Y2) * V_AB) * ones
            s32 = s23 = (K(R2, R3, Y2, Y3) * V_AB) * ones

            s = jnp.array([
                [s00, s01, s02, s03],
                [s10, s11, s12, s13],
                [s20, s21, s22, s23],
                [s30, s31, s32, s33]
            ]).transpose(2, 0, 1)

            return s                


class ShuntInductor(Model):
    """
    A model of a shunt inductor. 
    Internally uses Z-formulation to prevent divide-by-zero errors at L=0 or DC.

    Parameters:
    -----------
    L : Parameter
        The inductance in Henrys. Default is 1.0e-9 (1 nH).
    floating : bool
        If False (default), returns a standard 2-port model shunting to ground.
        If True, returns a 4-port differential model where the inductor bridges
        the top and bottom lines. Ports 0 & 1 act as the input pair, and Ports
        2 & 3 act as the output pair.
    """
    L: Parameter = 1e-9
    floating: bool = False

    def s(self, freq: Frequency) -> jnp.ndarray:
        w = freq.w
        L = self.L
        Z = 1j * w * L

        if not self.floating:        
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
        else:
            ones = jnp.ones(freq.npoints, dtype=jnp.complex128)
            if jnp.isscalar(self.z0):
                z0_0 = z0_1 = z0_2 = z0_3 = self.z0
            else:
                z0_0, z0_1 = self.z0[..., 0], self.z0[..., 1]
                z0_2, z0_3 = self.z0[..., 2], self.z0[..., 3]
            
            Y0, Y1 = 1.0 / z0_0, 1.0 / z0_1
            Y2, Y3 = 1.0 / z0_2, 1.0 / z0_3
            
            R0, R1 = z0_0.real, z0_1.real
            R2, R3 = z0_2.real, z0_3.real

            YA = Y0 + Y2
            YB = Y1 + Y3
            
            # Z-based Nodal Determinant
            D_Z = Z * YA * YB + YA + YB
            
            # Z-based Node voltage response factors
            V_AA = (Z * YB + 1.0) / D_Z
            V_BB = (Z * YA + 1.0) / D_Z
            V_AB = 1.0 / D_Z

            def K(r_i, r_j, y_i, y_j):
                return 2.0 * jnp.sqrt(r_i * r_j) * y_i * y_j

            s00 = (K(R0, R0, Y0, Y0) * V_AA - (jnp.conj(z0_0) / z0_0)) * ones
            s11 = (K(R1, R1, Y1, Y1) * V_BB - (jnp.conj(z0_1) / z0_1)) * ones
            s22 = (K(R2, R2, Y2, Y2) * V_AA - (jnp.conj(z0_2) / z0_2)) * ones
            s33 = (K(R3, R3, Y3, Y3) * V_BB - (jnp.conj(z0_3) / z0_3)) * ones

            s20 = s02 = (K(R0, R2, Y0, Y2) * V_AA) * ones
            s31 = s13 = (K(R1, R3, Y1, Y3) * V_BB) * ones

            s10 = s01 = (K(R0, R1, Y0, Y1) * V_AB) * ones
            s30 = s03 = (K(R0, R3, Y0, Y3) * V_AB) * ones
            s21 = s12 = (K(R1, R2, Y1, Y2) * V_AB) * ones
            s32 = s23 = (K(R2, R3, Y2, Y3) * V_AB) * ones

            s = jnp.array([
                [s00, s01, s02, s03],
                [s10, s11, s12, s13],
                [s20, s21, s22, s23],
                [s30, s31, s32, s33]
            ]).transpose(2, 0, 1)

            return s            

    
class InductorQ(Model):
    """
    A 2-port model of a series inductor with a finite Quality Factor (Q).

    Parameters:
    -----------
    L : Parameter
        The inductance in Henrys. Default is 1.0e-9 (1 nH).
    Q : Parameter
        The quality factor representing non-ideal losses. Default is 50.0.
    """
    L: Parameter = 1e-9
    Q: Parameter = 50.0

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

    Parameters:
    -----------
    C : Parameter
        The capacitance in Farads. Default is 1.0e-12 (1 pF).
    Q : Parameter
        The quality factor representing non-ideal losses. Default is 50.0.
    """
    C: Parameter = 1e-12
    Q: Parameter = 50.0

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
    
    
SHORT = Load(Fixed(-1.0))
OPEN = Load(Fixed(1.0))
MATCH = Load(Fixed(0.0))