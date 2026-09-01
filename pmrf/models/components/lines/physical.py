"""
Physical transmission lines (general, coaxial, microstrip, stripline)
"""
from scipy.constants import c
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.constraints import Positive, GreaterThan
from pmrf.materials import (
    AbstractConductor,
    AbstractDielectric,
    BulkConductor,
    ConstantDielectric,
    DielectricProperties,
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
    AbstractStriplineFormulation,
    CohnStriplineFormulation,
    KirschningJansenMicrostripDispersion,
    PlanarQuasiStaticResult,
    TescheCoaxialFormulation,
    WheelerMicrostripFormulation,
    _wheeler_conductor_loss_factor,
)


# `None` disables modal dispersion, so it cannot double as "not given".
_DEFAULT_DISPERSION = object()

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
        filling sets that material's ``mu_r``.
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
            dielectric=self.dielectric.properties(freq),
            conductor=self.conductor.properties(freq),
        )
    
    
class MicrostripLine(AbstractImmittanceLine):
    r"""
    Microstrip line defined by standard geometry and material modules.
    
    Uses :class:`WheelerMicrostripFormulation` for the default mathematical formulation.

    The quasi-static formulation returns a :class:`PlanarQuasiStaticResult`.
    :meth:`PlanarQuasiStaticResult.to_immittance` converts it directly whether
    modal dispersion is disabled or not: with dispersion enabled, the dispersed
    $(\varepsilon_e, Z_c)$ replace the quasi-static ones in a fresh
    :class:`PlanarQuasiStaticResult` first, so both paths report the same
    genuine RLGC $Z_c=\sqrt{Z/Y}$ rather than the dispersion path tautologically
    reproducing Kirschning-Jansen's modal $Z_c$.

    **Mathematical Formulation**

    The line evaluates material permittivity, a quasi-static formulation, and
    then the optional modal-dispersion formulation. ParamRF carries permittivity
    complex throughout, following the ADS/AWR convention, so the dielectric loss
    is carried directly by the effective permittivity and needs no separate
    attenuation term:
    $$\gamma_m=\frac{j\omega}{c}\sqrt{\varepsilon_e(f)}.$$

    Static bulk conductivity is not folded into that permittivity, which would
    make it singular at DC. It is applied separately as a shunt conductance
    $G = \sigma K_g$, where $K_g$ is the geometric
    ``shunt_conductance_factor`` of the quasi-static result, so the line keeps a
    finite, nonzero conductance down to DC.

    The substrate must be nonmagnetic. No cited microstrip formulation covers
    magnetic media, so $\mu_r \neq 1$ is rejected rather than silently ignored.

    Wheeler's incremental-inductance rule (:func:`_wheeler_conductor_loss_factor`)
    is the single conductor-loss term charged on both paths:
    $$\alpha_c=\frac{\Re(Z_s)}{\Re(Z_{c,loss})W}
    \exp\left[-1.2\left(\frac{\Re(Z_{c,loss})}{Z_0}\right)^{0.7}\right],$$
    applied unconditionally, over the physical width $W$ rather than any
    thickness-widened one. So ``dispersion=None`` is a pure dispersion toggle:
    the quasi-static formulation's own `conductor_loss_factor` charges the
    same rule at $Z_{c,loss}=Z_{c,0}$, and the dispersion path charges it
    again at the dispersed $Z_{c,loss}$. The rule contains no $t$: it sums
    over every receded conductor surface, and the broad-face terms do not
    vanish as $t\to0$. So $t=\text{None}$ ("thickness unspecified") is not
    the same input as $t=0$; it is read as skin effect being in operation
    regardless, which is the good-faith default since Wheeler's $R_s$ is
    itself a thick-conductor result. $t$ only refines the geometry (through
    the quasi-static formulation, where supported), it does not gate whether
    conductor loss is applied. This is a ParamRF convention, not a correction
    of Kirschning-Jansen: K-J's $Z_c$ is a power-current quasi-TEM modal
    quantity, chosen by Jansen & Koster for its weak frequency dependence, and
    NIST (Williams, Alpert, Arz et al., *Causal Characteristic Impedance of
    Planar Transmission Lines*) establishes that microstrip has no unique
    $Z_c$.

    The sheet model above gives $R\to0$ as $f\to0$, which is wrong for a
    trace of known thickness: a finite $t$ gets a dc floor
    $R_{dc}=\rho/(Wt)$, blended smoothly with the skin-effect term as
    $R=\sqrt{R_{dc}^2+R_{ac}^2}$ (see
    :meth:`~pmrf.models.components.lines.formulations.PlanarQuasiStaticResult.to_immittance`).
    $t=\text{None}$ gets no floor: it asserts skin effect in operation at
    every frequency including dc, so there is no dc regime for a floor to
    describe, matching ADS, which applies no floor at all. The blend itself
    is a ParamRF convention rather than a rule any cited source prescribes
    -- mcalc/wcalc hard-switch to a dc solution once skin depth exceeds
    thickness instead, an equally defensible alternative.

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
        Thickness of the conductor. ``None`` means the thickness is
        unspecified, not that it is zero: skin effect is assumed to be in
        operation regardless, and Wheeler's conductor-loss correction (which
        does not depend on `t`) is applied unconditionally. Wheeler's
        quasi-static formulation requires ``None``; thickness-aware
        formulations such as Hammerstad--Jensen use a positive value to
        refine the geometry.
    formulation : AbstractMicrostripFormulation, default=WheelerMicrostripFormulation()
        The closed-form physics used to compute the quasi-static solution.
    dispersion : AbstractMicrostripDispersion | None, default=KirschningJansenMicrostripDispersion()
        The modal-dispersion correction. ``None`` disables modal dispersion and
        preserves the quasi-static immittance path.

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
    dispersion: AbstractMicrostripDispersion | None = field(
        default_factory=KirschningJansenMicrostripDispersion
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
            KirschningJansenMicrostripDispersion()
            if dispersion is _DEFAULT_DISPERSION else dispersion
        )
        self.length = length
        self.name = name
        self.metadata = metadata

    def _resolved_quasi_static(self, freq: Frequency) -> PlanarQuasiStaticResult:
        """The quasi-static solution actually in force, dispersed when ``dispersion`` is set.

        Both :meth:`immittance` and the public :meth:`ep_eff`/:meth:`w_eff`
        accessors route through this single pipeline, so the accessors cannot
        drift from what ``immittance`` actually charges.
        """
        substrate = self.substrate
        dielectric = substrate.dielectric.properties(freq)
        ep_r = eqx.error_if(
            dielectric.ep_r,
            jnp.any(jnp.abs(dielectric.mu_r - 1) > 1e-12),
            "microstrip formulations require a nonmagnetic substrate (mu_r = 1)",
        )

        quasi_static = self.formulation.quasi_static(
            w=self.w,
            h=substrate.h,
            t=substrate.t,
            ep_r=ep_r,
        )

        if self.dispersion is None:
            return quasi_static

        ep_eff, zc = self.dispersion.disperse(
            freq,
            ep_eff_0=quasi_static.ep_eff,
            zc_0=quasi_static.zc,
            ep_r=ep_r,
            w_eff=quasi_static.w_eff,
            h=substrate.h,
        )

        # Route through the same to_immittance a quasi-static line uses, at
        # the dispersed (ep_eff, zc) rather than inverting the modal (zc,
        # gamma) through from_zc_gamma. That inversion made Zc = sqrt(Z/Y)
        # tautologically equal to the modal zc, so the conductor never
        # genuinely entered it; to_immittance instead builds Z and Y from
        # physical per-unit-length quantities, so Zc = sqrt(Z/Y) falls out
        # exact with the conductor contribution actually in it. This is a
        # ParamRF convention, not a correction of Kirschning-Jansen: K-J's Zc
        # is a power-current quasi-TEM modal quantity, chosen by Jansen &
        # Koster for its weak frequency dependence, and NIST (Williams,
        # Alpert, Arz et al., "Causal Characteristic Impedance of Planar
        # Transmission Lines") establishes that microstrip has no unique Zc.
        #
        # Wheeler's incremental-inductance rule applies at every thickness.
        # It contains no `t`: the broad-face terms it sums over do not vanish
        # as t -> 0, so t=None ("thickness unspecified") is not the same as
        # t=0. Skin effect is assumed to be in operation regardless. This is
        # the same conductor_loss_factor the quasi-static path charges,
        # evaluated here at the dispersed zc.
        conductor_loss_factor = _wheeler_conductor_loss_factor(self.w, zc)
        return PlanarQuasiStaticResult(
            ep_eff=ep_eff,
            zc=zc,
            w_eff=quasi_static.w_eff,
            conductor_loss_factor=conductor_loss_factor,
            shunt_conductance_factor=quasi_static.shunt_conductance_factor,
        )

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        substrate = self.substrate
        conductor = substrate.conductor.properties(freq)
        dielectric = substrate.dielectric.properties(freq)
        t = substrate.t
        # t=None asserts skin effect in operation at every frequency,
        # including dc, so there is no dc regime for a floor to describe; a
        # finite t gets R_dc = rho/(W*t) = 1/(sigma*W*t).
        r_dc = None if t is None else 1.0 / (conductor.sigma * self.w * t)
        return self._resolved_quasi_static(freq).to_immittance(
            freq, dielectric, conductor, r_dc=r_dc
        )

    def ep_eff(self, freq: Frequency) -> jnp.ndarray:
        r"""
        Complex effective relative permittivity the line actually ends up with.

        Dispersed via ``dispersion`` when it is set, quasi-static otherwise —
        the same value :meth:`immittance` uses internally, so it is exposed
        here rather than recomputed by the caller. The imaginary part carries
        the dielectric loss, following the ADS/AWR convention of carrying
        permittivity complex throughout (see the class docstring and #79).

        Parameters
        ----------
        freq : Frequency
            Frequencies at which to evaluate the line.

        Returns
        -------
        jnp.ndarray
            Complex effective relative permittivity, shape ``(npoints,)``.
        """
        return self._resolved_quasi_static(freq).ep_eff

    def w_eff(self, freq: Frequency) -> jnp.ndarray:
        r"""
        Effective conductor width the line actually ends up with.

        Dispersed via ``dispersion`` when it is set, quasi-static otherwise —
        the same value :meth:`immittance` uses internally, so it is exposed
        here rather than recomputed by the caller.

        Parameters
        ----------
        freq : Frequency
            Frequencies at which to evaluate the line.

        Returns
        -------
        jnp.ndarray
            Effective conductor width in meters, shape ``(npoints,)``.
        """
        return self._resolved_quasi_static(freq).w_eff


class StriplineLine(AbstractImmittanceLine):
    r"""
    Stripline defined by its geometry and material modules.

    Uses :class:`CohnStriplineFormulation` as the default mathematical
    formulation.

    Stripline is homogeneously filled, so it has no modal dispersion and
    therefore no ``dispersion`` field: the effective permittivity is the
    permittivity of the filling itself, at every frequency. Material dispersion
    still applies, and needs nothing stripline-specific — it arrives through the
    dielectric module exactly as it does for any other line.

    **Mathematical Formulation**

    The quasi-static formulation returns $(\varepsilon_e, Z_c, W_{eff})$, and
    :meth:`PlanarQuasiStaticResult.to_immittance` converts them directly:
    $$Z = \frac{j\omega Z_c\sqrt{\varepsilon_e}}{c} + \frac{2Z_s}{W_{eff}}
    \qquad
    Y = \frac{j\omega\sqrt{\varepsilon_e}}{Z_c c}.$$
    See :class:`CohnStriplineFormulation` for the geometry.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import StriplineLine
        from pmrf.materials import BulkConductor, ConstantDielectric

        line = StriplineLine(
            w=2.655e-3,
            b=3.2e-3,
            t=35e-6,
            dielectric=ConstantDielectric(ep_r=2.2, tand=0.001),
            conductor=BulkConductor(rho=1.72e-8),
            length=0.1,
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s = line.s(freq)

    Parameters
    ----------
    w : Param, default=2.655e-3
        Width of the centre strip in meters.
    b : Param, default=3.2e-3
        Separation of the ground planes in meters.
    t : Param | None, default=35e-6
        Thickness of the centre strip in meters. ``None`` idealises it as
        zero-thickness, which has no finite conductor loss.
    dielectric : AbstractDielectric, default=ConstantDielectric(ep_r=4.3)
        The filling between the ground planes. A scalar permittivity or an
        ``(ep_r, tand)`` tuple is coerced into a
        :class:`~pmrf.materials.ConstantDielectric`.
    conductor : AbstractConductor, default=BulkConductor()
        The material of the strip and the ground planes. A scalar resistivity in
        ohm-meters is coerced into a :class:`~pmrf.materials.BulkConductor`.
    formulation : AbstractStriplineFormulation, default=CohnStriplineFormulation()
        The closed-form physics used to compute the quasi-static solution.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE Transactions
    on Microwave Theory and Techniques, 3(2), 119-126.

    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.7. Wiley.
    """
    #: Width of the centre strip
    w: Param = param(default=2.655e-3, constraint=Positive())

    #: Separation of the ground planes
    b: Param = param(default=3.2e-3, constraint=Positive())

    #: Thickness of the centre strip
    t: Param | None = field(
        default=35e-6,
        converter=lambda x: as_param(x, constraint=Positive()) if x is not None else None,
    )

    #: The filling between the ground planes
    dielectric: AbstractDielectric = field(
        default_factory=lambda: ConstantDielectric(ep_r=4.3), converter=as_dielectric
    )

    #: The material of the strip and the ground planes
    conductor: AbstractConductor = field(
        default_factory=BulkConductor, converter=as_conductor
    )

    #: The underlying physics formulation
    formulation: AbstractStriplineFormulation = field(
        default_factory=CohnStriplineFormulation
    )

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        dielectric = self.dielectric.properties(freq)
        conductor = self.conductor.properties(freq)
        quasi_static = self.formulation.quasi_static(
            w=self.w,
            b=self.b,
            t=self.t,
            ep_r=dielectric.ep_r,
        )
        return quasi_static.to_immittance(freq, dielectric, conductor)
