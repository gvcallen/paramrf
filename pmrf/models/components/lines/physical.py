"""
Physical transmission lines (general, coaxial, microstrip)
"""
from scipy.constants import c
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.constraints import Positive, GreaterThan
from pmrf.materials import (
    AbstractConductor,
    AbstractDielectric,
    BulkConductor,
    ConstantDielectric,
    as_conductor,
    as_dielectric,
)
from pmrf.utils import field
from pmrf.parameters import Param, param, as_param
from pmrf.models.components.lines.base import AbstractImmittanceLine, ImmittanceResult
from pmrf.models.components.lines.formulations import (
    AbstractCoaxialFormulation,
    AbstractMicrostripFormulation,
    TescheCoaxialFormulation,
    WheelerMicrostripFormulation,
)

# -----------------------------------------------------------------------------
# Lines
# -----------------------------------------------------------------------------

class PhysicalLine(AbstractImmittanceLine):
    r"""
    Transmission line defined by nominal characteristic impedance, relative permittivity, 
    conductor attenuation, and dielectric loss tangent.
    
    **Mathematical Formulation**

    The frequency-dependent attenuation components are computed as:
    $$\alpha_c = A \cdot \sqrt{\frac{f}{f_A}} \cdot \frac{\ln(10)}{20}$$
    $$\alpha_d = \frac{\pi f \sqrt{\varepsilon_r}}{c} \cdot \tan\delta$$

    Which yield the per-unit-length parameters:
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
        A_dB = self.A * jnp.sqrt(f / self.f_A)

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
    Transmission line defined by common datasheet parameters (nominal impedance
    and velocity/loss factors). Includes skin effect (`k1`) and 
    dielectric loss (`k2`).

    **Mathematical Formulation**

    The normalized loss coefficients ($k_{1,norm}$, $k_{2,norm}$) depend on `loss_coeffs_normalized`. 
    Attenuation variables scale natively with $\sqrt{\omega}$ and $\omega$:
    $$\alpha_c = k_{1,norm} \cdot \frac{\ln(10)}{20} \cdot \sqrt{\omega}$$
    $$\alpha_d = k_{2,norm} \cdot \frac{\ln(10)}{20} \cdot \omega$$

    Resulting in the per-unit-length components:
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
        If True, k1 and k2 are evaluated directly without normalizing to 100MHz references.
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

        sqrt_w = jnp.sqrt(w)
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
    

class CoaxialLine(AbstractImmittanceLine):
    r"""
    Coaxial line defined directly by its physical geometry and material modules.
    
    Uses :class:`TescheCoaxialFormulation` as the default mathematical formulation.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import CoaxialLine
        from pmrf.materials import BulkConductor, ConstantDielectric

        phys_cable = CoaxialLine(
            d_in=0.9e-3,
            d_out=2.95e-3,
            dielectric=ConstantDielectric(ep_r=1.5, tand=0.0004),
            conductor=BulkConductor(rho=1.72e-8),
            length=0.5
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s_phys = phys_cable.s(freq)

    Parameters
    ----------
    d_in : Param, default=1.12e-3
        Inner conductor diameter in meters.
    d_out : Param, default=3.2e-3
        Outer conductor inner diameter in meters.
    dielectric : AbstractDielectric, default=ConstantDielectric()
        The dielectric filling. A scalar permittivity or an ``(ep_r, tand)`` tuple
        is coerced into a :class:`~pmrf.materials.ConstantDielectric`.
    mu_r : Param, default=1.0
        Relative permeability of the dielectric filling.
    conductor : AbstractConductor, default=BulkConductor()
        The conductor material of both conductors. A scalar resistivity in
        ohm-meters is coerced into a :class:`~pmrf.materials.BulkConductor`.
    formulation : AbstractCoaxialFormulation, default=TescheCoaxialFormulation()
        The closed-form physics used to compute the immittance.
    """
    #: Inner conductor diameter
    d_in: Param = param(default=1.12e-3, constraint=Positive())
    
    #: Outer conductor inner diameter
    d_out: Param = param(default=3.2e-3, constraint=Positive())
    
    #: The dielectric filling
    dielectric: AbstractDielectric = field(
        default_factory=ConstantDielectric, converter=as_dielectric
    )
    
    #: Relative permeability of the dielectric filling
    mu_r: Param = param(default=1.0, constraint=Positive())
    
    #: The conductor material of both conductors
    conductor: AbstractConductor = field(
        default_factory=BulkConductor, converter=as_conductor
    )
    
    #: The underlying physics formulation
    formulation: AbstractCoaxialFormulation = field(default_factory=TescheCoaxialFormulation)

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        return self.formulation.immittance(
            freq,
            d_in=self.d_in,
            d_out=self.d_out,
            ep_r=self.dielectric.epsilon_r(freq),
            mu_r=self.mu_r,
            conductor=self.conductor,
        )
    
    
class MicrostripLine(AbstractImmittanceLine):
    r"""
    Microstrip line defined by standard geometry and material modules.
    
    Uses :class:`WheelerMicrostripFormulation` for the default mathematical formulation.

    The formulation returns a :class:`QuasiStaticResult`, which
    :meth:`QuasiStaticResult.to_immittance` turns into an immittance using the
    conductor's surface impedance.
    
    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import MicrostripLine
        from pmrf.materials import BulkConductor, ConstantDielectric

        phys_microstrip = MicrostripLine(
            w=4e-3,
            h=2.0e-3,
            dielectric=ConstantDielectric(ep_r=4.6, tand=0.025),
            conductor=BulkConductor(rho=1.72e-8),
            length=0.5
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s_phys = phys_microstrip.s(freq)    

    Parameters
    ----------
    w : Param, default=3e-3
        Width of the microstrip trace in meters.
    h : Param, default=1.6e-3
        Height of the dielectric substrate in meters.
    dielectric : AbstractDielectric, default=ConstantDielectric(ep_r=4.3)
        The substrate material. A scalar permittivity or an ``(ep_r, tand)`` tuple
        is coerced into a :class:`~pmrf.materials.ConstantDielectric`.
    conductor : AbstractConductor, default=BulkConductor()
        The material of the trace and ground plane. A scalar resistivity in
        ohm-meters is coerced into a :class:`~pmrf.materials.BulkConductor`.
    t : Param | None, default=None
        Thickness of the conductor. Not yet used, provided for future compatibility.
    formulation : AbstractMicrostripFormulation, default=WheelerMicrostripFormulation()
        The closed-form physics used to compute the quasi-static solution.
    """
    #: Width of the microstrip trace
    w: Param = param(default=3e-3, constraint=Positive())
    
    #: Height of the dielectric substrate
    h: Param = param(default=1.6e-3, constraint=Positive())
    
    #: The substrate material
    dielectric: AbstractDielectric = field(
        default_factory=lambda: ConstantDielectric(ep_r=4.3), converter=as_dielectric
    )
    
    #: The material of the trace and ground plane
    conductor: AbstractConductor = field(
        default_factory=BulkConductor, converter=as_conductor
    )
    
    #: Thickness of the conductor. Not yet used, provided for future compatibility.
    t: Param | None = field(default=None, converter=lambda x: as_param(x, constraint=Positive()) if x is not None else None)
    
    #: The underlying physics formulation
    formulation: AbstractMicrostripFormulation = field(default_factory=WheelerMicrostripFormulation)
    
    def __post_init__(self):
        if self.t is not None:
            raise ValueError("Thickness not yet supported in `MicrostripLine`")

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        zs = self.conductor.surface_impedance(freq)

        quasi_static = self.formulation.quasi_static(
            freq,
            w=self.w,
            h=self.h,
            t=self.t,
            ep_r=self.dielectric.epsilon_r(freq),
            zs=zs,
        )

        return quasi_static.to_immittance(freq, zs)
