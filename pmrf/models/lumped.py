import jax.numpy as jnp

from pmrf.parameters import Parameter
from pmrf._model import Model
from pmrf._frequency import Frequency

class Load(Model):
    """
    **Overview**

    An abstract base class for all 1-port terminating load models.

    This class establishes the interface for 1-port networks, which are
    defined by their reflection coefficient, `gamma`. Subclasses must implement
    the `gamma` property.
    """
    nports: int = 1
    
    @property
    def gamma(self) -> float | jnp.ndarray:
        """The complex reflection coefficient (Gamma) of the load."""
        raise NotImplementedError("Subclasses of Load must implement the 'gamma' property.")

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
    
class Match(Load):
    """
    **Overview**

    An ideal matched load (perfect termination).
    """
    @property
    def gamma(self) -> float | jnp.ndarray:
        """The reflection coefficient of a matched load, which is always 0."""
        return 0.0

class Short(Load):
    """
    **Overview**

    An ideal short circuit.
    """
    @property
    def gamma(self) -> float | jnp.ndarray:
        """The reflection coefficient of a short circuit, which is always -1."""
        return -1.0

class Open(Load):
    """
    **Overview**

    An ideal open circuit.
    """
    @property
    def gamma(self) -> float | jnp.ndarray:
        """The reflection coefficient of an open circuit, which is always +1."""
        return 1.0
 
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
        
class Transformer(Model):
    """
    **Overview**

    An ideal, lossless, frequency-independent 4-port 1:1 transformer.
    """
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Returns the fixed S-parameter matrix for the transformer.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The 4x4 S-parameter matrix, constant across frequency.
        """
        s = 0.5 * jnp.ones((freq.npoints, 4, 4), dtype=jnp.complex128)
        s = s.at[:, 0, 3].set(-0.5)
        s = s.at[:, 1, 2].set(-0.5)
        s = s.at[:, 2, 1].set(-0.5)
        s = s.at[:, 3, 0].set(-0.5)

        return s