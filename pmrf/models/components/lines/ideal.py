"""
Ideal transmission lines (phase, constant RLGC)
"""
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.constraints import Positive
from pmrf.utils import field
from pmrf.parameters import Param, param
from pmrf.models.components.lines.base import TransmissionLine, RLGCLine, RLGCResult


class PhaseLine(TransmissionLine):
    r"""
    Ideal, lossless, and dispersionless transmission line defined by 
    electrical length at a reference frequency. Characteristic impedance 
    is real and constant; phase scales linearly.

    **Mathematical Formulation**

    $$Z_c(\omega) = Z_c$$
    $$\gamma L(\omega) = j \cdot \left(\theta \cdot \frac{\pi}{180}\right) \cdot \frac{\omega}{\omega_0}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import PhaseLine

        # Create an ideal 90-degree (quarter-wave) 50-ohm line at 1 GHz
        quarter_wave = PhaseLine(
            z0=50.0,
            theta=90.0,
            f0=1e9
        )

        freq = prf.Frequency(start=0.5, stop=1.5, npoints=101, unit='ghz')
        s = quarter_wave.s(freq)

    Parameters
    ----------
    z0 : Param, default=50.0
        Characteristic impedance in Ohms.
    theta : Param, default=90.0
        Electrical length (phase shift) in degrees at reference frequency `f0`.
    f0 : float
        Reference frequency in Hz for `theta`. Key-word only static argument.
    """
    #: Electrical length (phase shift)
    theta: Param = param(default=90.0, constraint=Positive())
    
    #: Characteristic impedance
    z0: Param = param(default=50.0, constraint=Positive())
    
    #: Reference frequency
    f0: float = field(static=True, kw_only=True)

    def zc_and_gammaL(self, frequency: Frequency) -> jnp.ndarray:
        z0 = self.z0 * jnp.ones(frequency.npoints, dtype=complex)
        theta_rad = self.theta * jnp.pi / 180.0
        w0 = 2 * jnp.pi * self.f0
        beta_L = theta_rad * (frequency.w / w0)
        gammaL = 1j * beta_L
        
        return z0, gammaL


class ConstantRLGCLine(RLGCLine):
    r"""
    Transmission line with constant, frequency-independent RLGC parameters.

    **Mathematical Formulation**

    $$R(\omega) = R$$
    $$L(\omega) = L$$
    $$G(\omega) = G$$
    $$C(\omega) = C$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import ConstantRLGCLine

        lossless_line = ConstantRLGCLine(
            L=368.8e-9,  # nH/m
            C=147.5e-12, # pF/m
            length=0.1   # 10 cm
        )

        freq = prf.Frequency(start=1, stop=5, npoints=101, unit='ghz')
        s = lossless_line.s(freq)

    Parameters
    ----------
    R : Param, default=0.0
        Resistance in Ohms/m.
    L : Param, default=280e-9
        Inductance in Henries/m.
    G : Param, default=0.0
        Conductance in Siemens/m.
    C : Param, default=90e-12
        Capacitance in Farads/m.
    """
    #: Resistance in Ohms/m
    R: Param = param(default=0.0, constraint=Positive())
    
    #: Inductance in Henries/m
    L: Param = param(default=280e-9, constraint=Positive())
    
    #: Conductance in Siemens/m
    G: Param = param(default=0.0, constraint=Positive())
    
    #: Capacitance in Farads/m
    C: Param = param(default=90e-12, constraint=Positive())

    def rlgc(self, freq: Frequency) -> RLGCResult:
        ones = jnp.ones(freq.npoints)
        R, L, G, C = self.R * ones, self.L * ones, self.G * ones, self.C * ones

        return RLGCResult(R=R, L=L, G=G, C=C)
    