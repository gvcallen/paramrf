"""Coaxial transmission-line models and formulations."""
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import epsilon_0, mu_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.materials import AbstractConductor, AbstractDielectric, BulkConductor, ConstantDielectric, ConductorProperties, DielectricProperties, as_conductor, as_dielectric
from pmrf.materials.surface_impedance import AbstractSurfaceImpedance, SchelkunoffRodSurfaceImpedance, SchelkunoffTubeSurfaceImpedance, TescheRodSurfaceImpedance, TescheTubeSurfaceImpedance
from pmrf.models.components.lines.base import AbstractImmittanceLine, ImmittanceResult
from pmrf.parameters import Param, as_param, param
from pmrf.utils import field

class AbstractCoaxialFormulation(eqx.Module):
    """Abstract base class for a closed-form coaxial formulation."""
    @abstractmethod
    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor: ConductorProperties | None = None, shield_thickness=None) -> ImmittanceResult:
        r"""
        Calculate the per-unit-length immittance.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        d_in : ArrayLike
            Inner conductor diameter in meters.
        d_out : ArrayLike
            Outer conductor inner diameter in meters.
        dielectric : DielectricProperties
            Evaluated relative permittivity and permeability.
        conductor : ConductorProperties
            Evaluated inner-conductor properties.
        outer_conductor : ConductorProperties | None, default=None
            Evaluated shield properties, or ``None`` to reuse ``conductor``.
        shield_thickness : ArrayLike | None, default=None
            Shield wall thickness, or ``None`` for an infinitely thick shield.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance.
        """
        raise NotImplementedError


def _coaxial_immittance(freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor: ConductorProperties | None, shield_thickness, inner_impedance: AbstractSurfaceImpedance, shield_impedance: AbstractSurfaceImpedance) -> ImmittanceResult:
    """Calculate coaxial immittance for the supplied surface impedances."""
    eps = epsilon_0 * dielectric.ep_r
    mu = mu_0 * dielectric.mu_r
    omega = freq.w

    d_out = eqx.error_if(d_out, d_out - d_in <= 0, "d_out must exceed d_in")
    a, b = d_in / 2, d_out / 2
    ln_b_over_a = jnp.log(b / a)

    L_ext = jnp.ones(freq.npoints) * mu / (2 * jnp.pi) * ln_b_over_a

    Z_int = inner_impedance.impedance(omega, conductor, a=a) / (2 * jnp.pi * a)
    # An unmodelled wall is an infinitely thick one, which every tube
    # formulation takes analytically -- so the shield is one call whatever
    # formulation it is, with no branch on its type or on whether a wall was given.
    shield_conductor = conductor if outer_conductor is None else outer_conductor
    wall = jnp.inf if shield_thickness is None else shield_thickness
    shield_zs = shield_impedance.impedance(omega, shield_conductor, a=b, t=wall)
    Z_int = Z_int + shield_zs / (2 * jnp.pi * b)

    Z = 1j * omega * L_ext + Z_int
    Y = 1j * omega * 2 * jnp.pi * eps / ln_b_over_a
    Y = Y + 2 * jnp.pi * dielectric.sigma / ln_b_over_a

    return ImmittanceResult(Z=Z, Y=Y, omega=omega)


class TescheCoaxialFormulation(AbstractCoaxialFormulation):
    r"""
    Coaxial line formulation using Tesche's equivalent-circuit approximation.

    **Mathematical Formulation**

    The external per-unit-length inductance and the shunt admittance follow from
    the coaxial geometry and the complex permittivity
    $\varepsilon = \varepsilon_0 \varepsilon_r$:
    $$L' = \frac{\mu_0 \mu_r}{2\pi} \ln\left(\frac{b}{a}\right)
    \qquad
    Y = \frac{j\omega 2\pi\varepsilon}{\ln(b/a)}$$

    Each conductor contributes $Z_s/(2\pi r)$. The defaults are
    :class:`~pmrf.materials.surface_impedance.TescheRodSurfaceImpedance` and
    :class:`~pmrf.materials.surface_impedance.TescheTubeSurfaceImpedance`. An unspecified
    shield thickness uses the infinite-wall limit. Complex $\varepsilon_r$
    and $\mu_r$ account for dielectric and magnetic loss.

    **Validity**

    The conductor circuit interpolates between dc resistance and half-space
    impedance but omits the exact $1/(2\gamma a)$ curvature term. Use
    :class:`SchelkunoffCoaxialFormulation` for the exact cylindrical solution.
    The TEM line model applies below the TE11 cutoff
    $f_c \approx c / \left[\pi (a + b) \sqrt{\varepsilon_r \mu_r}\right]$.

    References
    ----------
    Tesche, F. M. (2007). A Simple Model for the Line Parameters of a Lossy Coaxial
    Cable Filled With a Nondispersive Dielectric. IEEE Transactions on Electromagnetic
    Compatibility, 49(1), 12-17.

    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial Transmission Lines
    and Cylindrical Shields. Bell System Technical Journal, 13(4), 532-579.
    """
    #: Surface-impedance formulation for the inner conductor
    inner_impedance: AbstractSurfaceImpedance = TescheRodSurfaceImpedance()
    #: Surface-impedance formulation for the shield, called with the shield's inner
    #: radius and its wall thickness, infinite when none is given
    shield_impedance: AbstractSurfaceImpedance = TescheTubeSurfaceImpedance()

    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor=None, shield_thickness=None) -> ImmittanceResult:
        return _coaxial_immittance(
            freq, d_in=d_in, d_out=d_out, dielectric=dielectric,
            conductor=conductor, outer_conductor=outer_conductor,
            shield_thickness=shield_thickness, inner_impedance=self.inner_impedance,
            shield_impedance=self.shield_impedance,
        )


class SchelkunoffCoaxialFormulation(AbstractCoaxialFormulation):
    r"""
    Coaxial line formulation using Schelkunoff's exact cylindrical solution.

    **Mathematical Formulation**

    The external inductance and shunt admittance are
    $$L' = \frac{\mu_0 \mu_r}{2\pi} \ln\left(\frac{b}{a}\right)
    \qquad
    Y = \frac{j\omega 2\pi\varepsilon}{\ln(b/a)},$$
    The inner conductor uses
    :class:`~pmrf.materials.surface_impedance.SchelkunoffRodSurfaceImpedance` --
    Schelkunoff's eq. (65) rather than an equivalent circuit:
    $$Z_{inner} = \frac{\zeta_c}{2\pi a}\,
    \frac{I_0(\gamma a)}{I_1(\gamma a)},\qquad
    \gamma = \sqrt{j\omega\mu\sigma}.$$

    The shield uses Schelkunoff's tube solution, referred to its inner
    surface. An unspecified ``shield_thickness`` gives
    $Z_{outer}=\zeta_c K_0(\gamma b)/(2\pi bK_1(\gamma b))$. Both formulations
    are configurable fields.

    **Validity**

    Exact for homogeneous cylindrical conductors carrying axially symmetric
    current. The TEM line model applies below the TE11 cutoff
    $f_c \approx c / \left[\pi (a + b) \sqrt{\varepsilon_r \mu_r}\right]$.

    References
    ----------
    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial Transmission Lines
    and Cylindrical Shields. Bell System Technical Journal, 13(4), 532-579.
    Eq. (65), (74).
    """
    #: Surface-impedance formulation for the inner conductor
    inner_impedance: AbstractSurfaceImpedance = SchelkunoffRodSurfaceImpedance()
    #: Surface-impedance formulation for the shield, called with the shield's inner
    #: radius and its wall thickness, infinite when none is given
    shield_impedance: AbstractSurfaceImpedance = SchelkunoffTubeSurfaceImpedance()

    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor=None, shield_thickness=None) -> ImmittanceResult:
        return _coaxial_immittance(
            freq, d_in=d_in, d_out=d_out, dielectric=dielectric,
            conductor=conductor, outer_conductor=outer_conductor,
            shield_thickness=shield_thickness, inner_impedance=self.inner_impedance,
            shield_impedance=self.shield_impedance,
        )


class CoaxialLine(AbstractImmittanceLine):
    r"""
    Coaxial line defined by geometry and materials.
    
    The default :class:`SchelkunoffCoaxialFormulation` is the exact cylindrical
    solution. :class:`TescheCoaxialFormulation` provides a Bessel-free
    approximation.

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
            conductor=BulkConductor(sigma=5.8e7),
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
        The inner conductor material. A scalar conductivity in
        S/m is coerced into a :class:`~pmrf.materials.BulkConductor`.
    outer_conductor : AbstractConductor | None, default=None
        The shield material. ``None`` means the same material as ``conductor``.
    shield_thickness : Param | None, default=None
        Shield wall thickness in meters. ``None`` means an infinitely thick shield.
    formulation : AbstractCoaxialFormulation, default=SchelkunoffCoaxialFormulation()
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
    
    #: The inner conductor material
    conductor: AbstractConductor = field(
        default_factory=BulkConductor, converter=as_conductor
    )

    #: The optional shield material; None reuses the inner conductor
    outer_conductor: AbstractConductor | None = field(
        default=None,
        converter=lambda x: None if x is None else as_conductor(x),
    )

    #: The optional shield wall thickness in meters
    shield_thickness: Param | None = field(
        default=None,
        converter=lambda x: as_param(x, constraint=Positive()) if x is not None else None,
    )
    
    #: The underlying physics formulation
    formulation: AbstractCoaxialFormulation = field(default_factory=SchelkunoffCoaxialFormulation)

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        return self.formulation.immittance(
            freq,
            d_in=self.d_in,
            d_out=self.d_out,
            dielectric=self.dielectric.properties(freq),
            conductor=self.conductor.properties(freq),
            outer_conductor=(
                None if self.outer_conductor is None
                else self.outer_conductor.properties(freq)
            ),
            shield_thickness=(
                None if self.shield_thickness is None else self.shield_thickness
            ),
        )
