"""Coefficient-driven transmission-line models."""
import jax.numpy as jnp
from scipy.constants import c

from pmrf.constraints import GreaterThan, Positive
from pmrf.frequency import Frequency
from pmrf.models.components.lines.base import AbstractImmittanceLine, ImmittanceResult
from pmrf.parameters import Param, param
from pmrf.utils import field

class PhysicalLine(AbstractImmittanceLine):
    r"""
    Line defined by nominal impedance, permittivity, and loss coefficients.
    
    **Mathematical Formulation**

    The frequency-dependent attenuation components are computed as:
    $$\alpha_c = A \cdot \sqrt{\frac{f}{f_A}} \cdot \frac{\ln(10)}{20}$$
    $$\alpha_d = \frac{\pi f \sqrt{\varepsilon_r}}{c} \cdot \tan\delta$$

    The per-unit-length parameters are
    $$R = 2 z_n \alpha_c$$
    $$L = \frac{z_n \sqrt{\varepsilon_r}}{c}$$
    $$G = \frac{2 \alpha_d}{z_n}$$
    $$C = \frac{\sqrt{\varepsilon_r}}{z_n c}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import PhysicalLine

        line = PhysicalLine(
            zn=50.0,
            length=1.0,
            ep_r=2.2,
            A=0.01,
            f_A=1.0,
            tand=0.001
        )

        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        s = line.s(freq)

    Parameters
    ----------
    zn : Param, default=50.0
        Nominal characteristic impedance defining the L/C ratio.
    ep_r : Param, default=1.0
        Relative permittivity.
    A : Param, default=0.0
        Conductor loss in dB/m/sqrt(Hz).
    f_A : Param, default=1.0
        Frequency scaling reference for attenuation in Hz.
    tand : Param, default=0.0
        Dielectric loss tangent.
    """
    #: Nominal characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())
    
    #: Relative permittivity
    ep_r: Param = param(default=1.0, constraint=GreaterThan(1.0))
    
    #: Conductor loss in dB/m/sqrt(Hz)
    A: Param = param(default=0.0, constraint=Positive())
    
    #: Frequency scaling reference
    f_A: Param = param(default=1.0, constraint=Positive())
    
    #: Dielectric loss tangent
    tand: Param = param(default=0.0, constraint=Positive())

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        f = freq.f
        sqrt_ep_r = jnp.sqrt(self.ep_r)
        # sqrt is a branch point at f = 0. The attenuation is zero there, but
        # its raw gradient is not, so use the double-`where` pattern.
        safe_f = jnp.where(f > 0, f, 1.0)
        A_dB = self.A * jnp.where(f > 0, jnp.sqrt(safe_f / self.f_A), 0.0)

        alpha_c = A_dB * (jnp.log(10) / 20.0)
        alpha_d = jnp.pi * sqrt_ep_r * f / c * self.tand

        R_val = 2 * self.zn * alpha_c
        L_val = (self.zn * sqrt_ep_r) / c
        G_val = 2 / self.zn * alpha_d
        C_val = sqrt_ep_r / (self.zn * c)
        
        ones = jnp.ones(freq.npoints)
        R = R_val * ones
        L = L_val * ones
        G = G_val * ones
        C = C_val * ones
        
        return ImmittanceResult.from_rlgc(R, L, G, C, freq.w)
    

class DatasheetLine(AbstractImmittanceLine):
    r"""
    Line defined by nominal impedance, velocity factor, and loss coefficients.

    **Mathematical Formulation**

    With normalized coefficients $k_{1,norm}$ and $k_{2,norm}$,
    $$\alpha_c = k_{1,norm} \cdot \frac{\ln(10)}{20} \cdot \sqrt{\omega}$$
    $$\alpha_d = k_{2,norm} \cdot \frac{\ln(10)}{20} \cdot \omega$$

    and the per-unit-length parameters are
    $$R = 2 z_n \alpha_c$$
    $$L = \frac{z_n}{v_f c}$$
    $$G = \frac{2 \alpha_d}{z_n}$$
    $$C = \frac{1}{z_n v_f c}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import DatasheetLine

        cable = DatasheetLine(
            zn=50.0,
            vf=0.69,  # Velocity factor (e.g., solid PTFE)
            k1=0.2,   # Skin effect loss factor
            k2=0.01,  # Dielectric loss factor
            length=1.0
        )

        freq = prf.Frequency(start=0.1, stop=10, npoints=201, unit='ghz')
        s = cable.s(freq)

    Parameters
    ----------
    zn : Param, default=50.0
        Nominal characteristic impedance.
    vf : Param, default=1.0
        Velocity factor (ratio of propagation speed to the speed of light).
    k1 : Param, default=0.0
        Skin effect loss factor.
    k2 : Param, default=0.0
        Dielectric loss factor.
    loss_coeffs_normalized : bool, default=False
        If true, use ``k1`` and ``k2`` directly. Otherwise, normalize them to
        the 100 MHz reference convention.
    """
    #: Nominal characteristic impedance
    zn: Param = param(default=50.0, constraint=Positive())
    
    #: Velocity factor
    vf: Param = param(default=1.0, constraint=Positive())
    
    #: Skin effect loss factor
    k1: Param = param(default=0.0, constraint=Positive())
    
    #: Dielectric loss factor
    k2: Param = param(default=0.0, constraint=Positive())
    
    #: Loss coefficients normalization flag
    loss_coeffs_normalized: bool = field(default=False, static=True)

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        w = freq.w
        zn, k1, k2, vf = self.zn, self.k1, self.k2, self.vf

        if not self.loss_coeffs_normalized:
            k1_norm = k1 * (1.0 / (100 * jnp.sqrt(2 * jnp.pi * 10**6)))
            k2_norm = k2 * (1.0 / (100 * 2 * jnp.pi * 10**6))
        else:
            k1_norm = k1
            k2_norm = k2

        # sqrt is a branch point at w = 0, so guard it with the double-`where`
        # pattern: the conductor attenuation is zero at DC, its gradient is not.
        safe_w = jnp.where(w > 0, w, 1.0)
        sqrt_w = jnp.where(w > 0, jnp.sqrt(safe_w), 0.0)
        dBtoNeper = jnp.log(10) / 20
        alpha_c = k1_norm * dBtoNeper * sqrt_w
        alpha_d = k2_norm * dBtoNeper * w
        
        R = 2 * zn * alpha_c
        G = (2 / zn) * alpha_d
        
        # Broadcast L and C to the same shape as frequency arrays 
        # (w) in case zn and vf are provided as scalars.
        L = (zn / (vf * c)) * jnp.ones_like(w)
        C = (1.0 / (zn * vf * c)) * jnp.ones_like(w)
        
        return ImmittanceResult.from_rlgc(R, L, G, C, w)

