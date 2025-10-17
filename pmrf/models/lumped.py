import jax.numpy as jnp

from pmrf.parameters import Parameter
from pmrf.models.model import Model
from pmrf.frequency import Frequency

class Load(Model):
    """
    **Overview**

    An abstract base class for N-port loads defined by their reflection coefficient.
    """
    gamma: float
    nports: int = 1
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the 1-port S-parameter matrix.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The resultant 1x1 S-parameter matrix.
        """
        gamma, nports = self.gamma, self.nports
        # Create a frequency-dependent 1x1 matrix from the scalar gamma
        s = jnp.array(gamma).reshape(-1, 1, 1) * \
            jnp.eye(nports, dtype=jnp.complex128).reshape((-1, nports, nports)).\
            repeat(freq.npoints, 0)
        return s
 
class Capacitor(Model):
    """
    **Overview**

    A 2-port model of a series capacitor.

    **Example**

    ```python
    import pmrf as prf

    # Create a 1 pF series capacitor
    c_series = prf.models.Capacitor(C=1e-12)

    # Terminate the capacitor in a short to create a 1-port shunt capacitor
    c_shunt = c_series.terminated(prf.models.Short())

    freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
    s = c_shunt.s(freq)

    print(f"Shunt capacitor has {c_shunt.nports} port.")
    ```
    """
    C: Parameter = 1.0

    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the S-parameters of a series capacitor.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The resultant 2x2 S-parameter matrix.
        """
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
    **Overview**

    A 2-port model of a series inductor.

    **Example**

    ```python
    import pmrf as prf

    # Create a 1 nH series inductor
    l_series = prf.models.Inductor(L=1e-9)

    freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
    s = l_series.s(freq)

    print(f"Series inductor has {l_series.nports} ports.")
    ```
    """
    L: Parameter = 1.0
    
    def __post_init__(self):
        self.name = 'inductor'

    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the S-parameters of a series inductor.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The resultant 2x2 S-parameter matrix.
        """
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

    **Example**

    ```python
    import pmrf as prf

    # Create a 25-ohm series resistor (e.g., for an attenuator pad)
    r_series = prf.models.Resistor(R=25.0)

    freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
    s = r_series.s(freq)

    print(f"Series resistor has {r_series.nports} ports.")
    ```
    """
    R: Parameter = 1.0
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the S-parameters of a series resistor.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The resultant 2x2 S-parameter matrix.
        """
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
    **Overview**

    A 2-port model of a shunt capacitor.

    This model represents a capacitor connected from the signal path to the ground
    reference, placed between port 1 and port 2.

    **Example**

    ```python
    import pmrf as prf
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a 1 pF shunt capacitor
    c_shunt = prf.models.ShuntCapacitor(C=1e-12)

    # Define the frequency range for analysis
    freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
    
    # Calculate the S-parameters
    s = c_shunt.s(freq)
    ```
    """
    C: Parameter = 1.0

    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the S-parameters of a 2-port shunt capacitor.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            jnp.ndarray: The resultant 2x2 S-parameter matrix.
        """
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
    **Overview**

    A 2-port model of a shunt resistor.

    This model represents a resistor connected from the signal path to the ground
    reference, placed between port 1 and port 2.

    **Example**

    ```python
    import pmrf as prf
    import matplotlib.pyplot as plt

    # Create a 50 Ohm shunt resistor
    r_shunt = prf.models.ShuntResistor(R=50.0)

    # Define the frequency range for analysis
    freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
    
    # Calculate the S-parameters
    s = r_shunt.s(freq)

    # Plot the magnitude of S11 and S21 in dB
    s11_db = 20 * np.log10(np.abs(s[:, 0, 0]))
    s21_db = 20 * np.log10(np.abs(s[:, 1, 0]))

    plt.figure()
    plt.plot(freq.f_ghz, s11_db, label='S11 (Return Loss)')
    plt.plot(freq.f_ghz, s21_db, label='S21 (Insertion Loss)')
    plt.title('S-Parameters of a 50 Ohm 2-Port Shunt Resistor')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()
    ```
    """
    nports = 2
    R: Parameter = 50.0

    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the S-parameters of a 2-port shunt resistor.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            jnp.ndarray: The resultant 2x2 S-parameter matrix.
        """
        # Resistance value
        R = self.R
        
        # Port characteristic impedance
        z0 = self.z0

        # The admittance of the resistor is Y = 1/R
        # The normalized admittance is Yn = Y * Z0
        yn = z0 / R

        # Denominator for S-parameter conversion
        denom = 2.0 + yn

        # Calculate S-parameters
        s11 = -yn / denom
        s22 = s11
        s21 = 2.0 / denom
        s12 = s21

        # Construct the S-parameter matrix
        # We need to broadcast yn to match the shape of freq
        s11_freq = jnp.full(freq.w.shape, s11)
        s21_freq = jnp.full(freq.w.shape, s21)

        s = jnp.array([
            [s11_freq, s21_freq],
            [s21_freq, s11_freq]
        ]).transpose(2, 0, 1)

        return s

class ShuntInductor(Model):
    """
    **Overview**

    A 2-port model of a shunt inductor.

    This model represents an inductor connected from the signal path to the ground
    reference, placed between port 1 and port 2.

    **Example**

    ```python
    import pmrf as prf
    import matplotlib.pyplot as plt

    # Create a 1 nH shunt inductor
    l_shunt = prf.models.ShuntInductor(L=1e-9)

    # Define the frequency range for analysis
    freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
    
    # Calculate the S-parameters
    s = l_shunt.s(freq)

    # Plot the magnitude of S11 and S21 in dB
    s11_db = 20 * np.log10(np.abs(s[:, 0, 0]))
    s21_db = 20 * np.log10(np.abs(s[:, 1, 0]))

    plt.figure()
    plt.plot(freq.f_ghz, s11_db, label='S11 (Return Loss)')
    plt.plot(freq.f_ghz, s21_db, label='S21 (Insertion Loss)')
    plt.title('S-Parameters of a 1 nH 2-Port Shunt Inductor')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()
    ```
    """
    nports = 2
    L: Parameter = 1e-9

    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the S-parameters of a 2-port shunt inductor.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            jnp.ndarray: The resultant 2x2 S-parameter matrix.
        """
        # Angular frequency
        w = freq.w
        
        # Inductance value
        L = self.L
        
        # Port characteristic impedance
        z0 = self.z0

        # The admittance of the inductor is Y = 1 / (j*w*L)
        # The normalized admittance is Yn = Y * Z0
        yn = z0 / (1j * w * L)

        # Denominator for S-parameter conversion
        denom = 2.0 + yn

        # Calculate S-parameters
        s11 = -yn / denom
        s22 = s11
        s21 = 2.0 / denom
        s12 = s21

        # Construct the S-parameter matrix
        s = jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
    
SHORT = Load(-1.0)
OPEN = Load(1.0)
MATCH = Load(0.0)