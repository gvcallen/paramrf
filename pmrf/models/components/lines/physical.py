"""
Physical transmission lines (general, coaxial, microstrip)
"""
from typing import Literal

from scipy.constants import c, epsilon_0, mu_0
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.constraints import Positive, GreaterThan
from pmrf.materials import (
    AbstractConductor,
    AbstractDielectric,
    BulkConductor,
    ConstantDielectric,
    Substrate,
    as_conductor,
    as_dielectric,
    as_substrate,
)
from pmrf.utils import field
from pmrf.parameters import Param, param, as_param
from pmrf.models.components.lines.base import AbstractImmittanceLine, ImmittanceResult
from pmrf.models.components.lines.formulations import (
    AbstractCoaxialFormulation,
    AbstractMicrostripDispersion,
    AbstractMicrostripFormulation,
    ConductorProperties,
    DielectricProperties,
    KirschningJansen,
    TescheCoaxialFormulation,
    WheelerMicrostripFormulation,
)


EpsilonConvention = Literal["complex", "real"]

# `None` disables modal dispersion, so it cannot double as "not given".
_DEFAULT_DISPERSION = object()


def _as_epsilon_convention(value: str) -> EpsilonConvention:
    if value not in ("complex", "real"):
        raise ValueError("epsilon_convention must be 'complex' or 'real'")
    return value

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
        The dielectric filling, which supplies both its permittivity and its
        permeability. A scalar permittivity or an ``(ep_r, tand)`` tuple is
        coerced into a :class:`~pmrf.materials.ConstantDielectric`; a magnetic
        filling sets that material's ``mu_rel``.
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
            dielectric=DielectricProperties(
                self.dielectric.epsilon_r(freq), self.dielectric.mu_r(freq)
            ),
            conductor=ConductorProperties(
                self.conductor.surface_impedance(freq),
                self.conductor.sigma(freq),
                self.conductor.mu_r(freq),
            ),
        )
    
    
class MicrostripLine(AbstractImmittanceLine):
    r"""
    Microstrip line defined by standard geometry and material modules.
    
    Uses :class:`WheelerMicrostripFormulation` for the default mathematical formulation.

    The quasi-static formulation returns a :class:`QuasiStaticResult`. With
    modal dispersion disabled, :meth:`QuasiStaticResult.to_immittance` converts
    it directly; otherwise the dispersed modal quantities are inverted exactly.

    **Mathematical Formulation**

    The line evaluates material permittivity, a quasi-static formulation, and
    then the optional modal-dispersion formulation. Under the complex
    convention the material loss is carried directly by effective permittivity:
    $$\gamma_m=\frac{j\omega}{c}\sqrt{\varepsilon_e(f)}.$$
    Under the real convention, QUCS-like dielectric attenuation is added to a
    real phase constant:
    $$\alpha_d=\frac{\pi f}{c}\frac{\varepsilon_r}{\varepsilon_r-1}
    \frac{\varepsilon_e(0)-1}{\sqrt{\varepsilon_e(0)}}\tan\delta,
    \qquad \beta=\frac{\omega}{c}\sqrt{\varepsilon_e(f)}.$$

    For a finite-thickness conductor, Wheeler's incremental-inductance
    correction adds
    $$\alpha_c=\frac{\Re(Z_s)}{\Re(Z_{c,loss})W}
    \exp\left[-1.2\left(\frac{\Re(Z_{c,loss})}{Z_0}\right)^{0.7}\right].$$
    The resulting modal quantities are inverted exactly into the line's
    internal currency:
    $$Z=\gamma Z_c,\qquad Y=\frac{\gamma}{Z_c}.$$
    
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

    The substrate can be given as one :class:`~pmrf.materials.Substrate`, or as
    its loose fields. Both build the same canonical substrate, so the two
    idioms produce identical PyTrees; they may not be mixed in one call.

    Parameters
    ----------
    w : Param, default=3e-3
        Width of the microstrip trace in meters.
    substrate : Substrate, optional
        The substrate carrying the trace, as a single grouped module. Sharing
        one substrate between several traces is what dedupes their permittivity
        into a single parameter; see :class:`~pmrf.materials.Substrate`.
    h : Param, default=1.6e-3
        Height of the dielectric substrate in meters. Loose form of
        ``substrate.h``.
    dielectric : AbstractDielectric, default=ConstantDielectric(ep_r=4.3)
        The substrate material. A scalar permittivity or an ``(ep_r, tand)`` tuple
        is coerced into a :class:`~pmrf.materials.ConstantDielectric`.
    conductor : AbstractConductor, default=BulkConductor()
        The material of the trace and ground plane. A scalar resistivity in
        ohm-meters is coerced into a :class:`~pmrf.materials.BulkConductor`.
    t : Param | None, default=None
        Thickness of the conductor. Wheeler requires ``None``; thickness-aware
        formulations such as Hammerstad--Jensen use a positive value.
    formulation : AbstractMicrostripFormulation, default=WheelerMicrostripFormulation()
        The closed-form physics used to compute the quasi-static solution.
    dispersion : AbstractMicrostripDispersion | None, default=KirschningJansen()
        The modal-dispersion correction. ``None`` disables modal dispersion and
        preserves the quasi-static immittance path.
    epsilon_convention : {'complex', 'real'}, default='complex'
        Whether the quasi-static and modal formulas consume the full complex
        material permittivity or only its real part. The real convention adds
        dielectric attenuation separately.

    References
    ----------
    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.

    Kirschning, M., & Jansen, R. H. (1982). Accurate Model for Effective
    Dielectric Constant of Microstrip with Validity up to Millimeter-Wave
    Frequencies. Electronics Letters, 18(6), 272-273.

    Jansen, R. H., & Kirschning, M. (1983). Arguments and an Accurate Model for
    the Power-Current Formulation of Microstrip Characteristic Impedance.
    Archiv fuer Elektronik und Uebertragungstechnik, 37, 108-112.
    """
    #: Width of the microstrip trace
    w: Param = param(default=3e-3, constraint=Positive())

    #: The substrate carrying the trace
    substrate: Substrate = field(default_factory=Substrate, converter=as_substrate)

    #: The underlying physics formulation
    formulation: AbstractMicrostripFormulation = field(default_factory=WheelerMicrostripFormulation)

    #: The modal-dispersion formulation, or None to disable it
    dispersion: AbstractMicrostripDispersion | None = field(default_factory=KirschningJansen)

    #: Permittivity convention used by the empirical formulas
    epsilon_convention: EpsilonConvention = field(
        default="complex", static=True, converter=_as_epsilon_convention
    )

    def __init__(
        self,
        w: Param = 3e-3,
        *,
        length: Param,
        substrate: Substrate | None = None,
        h: Param | None = None,
        dielectric=None,
        conductor=None,
        t: Param | None = None,
        formulation: AbstractMicrostripFormulation | None = None,
        dispersion: AbstractMicrostripDispersion | None = _DEFAULT_DISPERSION,
        epsilon_convention: EpsilonConvention = "complex",
        name: str | None = None,
        metadata=None,
    ):
        loose = {"h": h, "dielectric": dielectric, "conductor": conductor, "t": t}
        given = {key: value for key, value in loose.items() if value is not None}
        if substrate is not None and given:
            raise ValueError(
                "pass either substrate= or the loose substrate fields "
                f"({', '.join(sorted(given))}), not both"
            )

        self.w = w
        self.substrate = Substrate(**given) if substrate is None else substrate
        self.formulation = (
            WheelerMicrostripFormulation() if formulation is None else formulation
        )
        self.dispersion = (
            KirschningJansen() if dispersion is _DEFAULT_DISPERSION else dispersion
        )
        self.epsilon_convention = epsilon_convention
        self.length = length
        self.name = name
        self.metadata = metadata

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        substrate = self.substrate
        zs = substrate.conductor.surface_impedance(freq)
        material_ep_r = substrate.dielectric.epsilon_r(freq)
        formula_ep_r = (
            material_ep_r if self.epsilon_convention == "complex" else jnp.real(material_ep_r)
        )

        quasi_static = self.formulation.quasi_static(
            freq,
            w=self.w,
            h=substrate.h,
            t=substrate.t,
            ep_r=formula_ep_r,
            zs=zs,
        )

        if self.dispersion is None and self.epsilon_convention == "complex":
            return quasi_static.to_immittance(freq, zs)

        if self.dispersion is None:
            ep_eff, zc = quasi_static.ep_eff, quasi_static.zc
        else:
            ep_eff, zc = self.dispersion.disperse(
                freq,
                ep_eff_0=quasi_static.ep_eff,
                zc_0=quasi_static.zc,
                ep_r=formula_ep_r,
                w=self.w,
                w_eff=quasi_static.w_eff,
                h=substrate.h,
                t=substrate.t,
            )

        if self.epsilon_convention == "complex":
            gamma = 1j * freq.w * jnp.sqrt(ep_eff) / c
        else:
            gamma = self._real_convention_gamma(
                freq, material_ep_r, ep_eff, quasi_static.ep_eff
            )

        # Wheeler's incremental-inductance rule. With no finite conductor
        # thickness scikit-rf defines this empirical correction as zero.
        if substrate.t is not None:
            z0 = jnp.sqrt(mu_0 / epsilon_0)
            loss_zc = zc if self.epsilon_convention == "complex" else quasi_static.zc
            current_distribution = jnp.exp(-1.2 * (jnp.real(loss_zc) / z0) ** 0.7)
            gamma = gamma + (
                jnp.real(zs) / (jnp.real(loss_zc) * self.w) * current_distribution
            )

        return ImmittanceResult.from_zc_gamma(zc, gamma, freq.w)

    @staticmethod
    def _real_convention_gamma(freq, material_ep_r, ep_eff, loss_ep_eff):
        ep_r_real = jnp.real(material_ep_r)
        ep_eff_real = jnp.real(ep_eff)
        loss_ep_eff_real = jnp.real(loss_ep_eff)
        tan_delta = -jnp.imag(material_ep_r) / ep_r_real

        delta_ep_r = ep_r_real - 1
        safe_delta = jnp.where(jnp.abs(delta_ep_r) > 1e-14, delta_ep_r, 1.0)
        filling = jnp.where(
            jnp.abs(delta_ep_r) > 1e-14,
            (loss_ep_eff_real - 1) / safe_delta,
            0.0,
        )
        alpha_dielectric = (
            jnp.pi
            * ep_r_real
            * filling
            / jnp.sqrt(loss_ep_eff_real)
            * tan_delta
            * freq.f
            / c
        )
        beta = freq.w * jnp.sqrt(ep_eff_real) / c
        return alpha_dielectric + 1j * beta
