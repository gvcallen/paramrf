"""Closed-form transmission-line formulations.

Formulations produce immittance or a quasi-static state. Dispersion
formulations add modal frequency dependence. Conductor shapes and current
distributions supply conductor surface impedance and geometry weights.
Material roughness is handled by the conductor material.
"""
from abc import abstractmethod
from scipy.constants import c, mu_0, epsilon_0
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.materials import ConductorProperties, DielectricProperties
from pmrf.materials.conductor_shape import (
    AbstractConductorShape,
    SchelkunoffRodShape,
    SchelkunoffTubeShape,
    TescheRodShape,
    TescheTubeShape,
)
from pmrf.models.components.lines.cross_section import AbstractPlanarCrossSection
from pmrf.models.components.lines.current_distribution import (
    AbstractCurrentDistribution,
    CohnCurrentDistribution,
    WheelerCurrentDistribution,
)
from pmrf.models.components.lines.base import ImmittanceResult


class PlanarQuasiStaticResult(eqx.Module):
    r"""
    Quasi-static solution of a single-conductor planar line over a ground plane.

    Contains the effective permittivity, characteristic impedance, effective
    conductor width, and static-conductivity geometry factor.

    Parameters
    ----------
    ep_eff : jnp.ndarray
        Complex effective relative permittivity, shape ``(npoints,)``.
    zc : jnp.ndarray
        Quasi-static characteristic impedance in ohms, $Z_a/\sqrt{\varepsilon_e}$.
    w_eff : jnp.ndarray
        Electromagnetic effective conductor width in meters.
    shunt_conductance_factor : jnp.ndarray
        Geometry factor multiplying static conductivity, in meters.
    """
    #: Complex effective relative permittivity
    ep_eff: jnp.ndarray

    #: Quasi-static characteristic impedance in ohms
    zc: jnp.ndarray

    #: Effective conductor width in meters
    w_eff: jnp.ndarray

    #: Geometry factor multiplying static conductivity
    shunt_conductance_factor: jnp.ndarray

    def to_immittance(
        self, freq: Frequency, dielectric: DielectricProperties,
        conductor: ConductorProperties,
        current_distribution: AbstractCurrentDistribution,
        cross_section: AbstractPlanarCrossSection,
    ) -> ImmittanceResult:
        r"""
        Convert the quasi-static solution to per-unit-length immittance.

        The external inductance and the shunt admittance follow from the
        quasi-static impedance and effective permittivity. Surface impedance
        is charged through the supplied current-distribution strategy:
        $$Z = \frac{j\omega Z_c \sqrt{\varepsilon_e}}{c} + Z_s K_c
        \qquad
        Y = \frac{j\omega \sqrt{\varepsilon_e}}{Z_c c}$$

        The current distribution supplies conductor shapes and geometry
        weights. Complex $\varepsilon_e$ contributes dielectric loss through
        the real part of $Y$.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        dielectric : DielectricProperties
            Evaluated relative permittivity and permeability.
        conductor : ConductorProperties
            Evaluated conductor properties.
        current_distribution : AbstractCurrentDistribution
            The strategy that charges surface impedance into $Z$. It must be
            written for the family ``cross_section`` belongs to.
        cross_section : AbstractPlanarCrossSection
            The line's frozen cross-section record.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance.

        References
        ----------
        Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.8. Wiley.
        """
        omega = freq.w
        sqrt_ep_eff_mu = jnp.sqrt(self.ep_eff * dielectric.mu_r)
        sqrt_ep_eff_over_mu = jnp.sqrt(self.ep_eff / dielectric.mu_r)

        Z = 1j * omega * self.zc * sqrt_ep_eff_mu / c
        # Cross-section dimensions reach the shape from the typed record, and
        # the weight the shape is about to be multiplied by travels with
        # them: an entry whose dc floor is fixed in per-unit-length terms
        # needs it to express that floor in this caller's normalisation.
        # Every other entry ignores it.
        dimensions = cross_section.dimensions()
        Z_cond = sum(
            shape.impedance(omega, conductor, weight=weight, **dimensions) * weight
            for shape, weight in current_distribution.distribute(
                freq, cross_section, self
            )
        )
        Z = Z + Z_cond
        Y = 1j * omega * sqrt_ep_eff_over_mu / (self.zc * c)
        Y = Y + dielectric.sigma * self.shunt_conductance_factor

        return ImmittanceResult(Z=Z, Y=Y, omega=omega)


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


def _coaxial_immittance(freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor: ConductorProperties | None, shield_thickness, inner_shape: AbstractConductorShape, shield_shape: AbstractConductorShape) -> ImmittanceResult:
    """Calculate coaxial immittance for the supplied conductor shapes."""
    eps = epsilon_0 * dielectric.ep_r
    mu = mu_0 * dielectric.mu_r
    omega = freq.w

    d_out = eqx.error_if(d_out, d_out - d_in <= 0, "d_out must exceed d_in")
    a, b = d_in / 2, d_out / 2
    ln_b_over_a = jnp.log(b / a)

    L_ext = jnp.ones(freq.npoints) * mu / (2 * jnp.pi) * ln_b_over_a

    Z_int = inner_shape.impedance(omega, conductor, a=a) / (2 * jnp.pi * a)
    # An unmodelled wall is an infinitely thick one, which every tube shape
    # takes analytically -- so the shield is one call whatever shape it is,
    # with no branch on the shape's type or on whether a wall was given.
    shield_conductor = conductor if outer_conductor is None else outer_conductor
    wall = jnp.inf if shield_thickness is None else shield_thickness
    shield_zs = shield_shape.impedance(omega, shield_conductor, a=b, t=wall)
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
    :class:`~pmrf.materials.conductor_shape.TescheRodShape` and
    :class:`~pmrf.materials.conductor_shape.TescheTubeShape`. An unspecified
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
    #: Cross-section shape of the inner conductor
    inner_shape: AbstractConductorShape = TescheRodShape()
    #: Cross-section shape of the shield, called with the shield's inner
    #: radius and its wall thickness, infinite when none is given
    shield_shape: AbstractConductorShape = TescheTubeShape()

    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor=None, shield_thickness=None) -> ImmittanceResult:
        return _coaxial_immittance(
            freq, d_in=d_in, d_out=d_out, dielectric=dielectric,
            conductor=conductor, outer_conductor=outer_conductor,
            shield_thickness=shield_thickness, inner_shape=self.inner_shape,
            shield_shape=self.shield_shape,
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
    :class:`~pmrf.materials.conductor_shape.SchelkunoffRodShape` --
    Schelkunoff's eq. (65) rather than an equivalent circuit:
    $$Z_{inner} = \frac{\zeta_c}{2\pi a}\,
    \frac{I_0(\gamma a)}{I_1(\gamma a)},\qquad
    \gamma = \sqrt{j\omega\mu\sigma}.$$

    The shield uses Schelkunoff's tube solution, referred to its inner
    surface. An unspecified ``shield_thickness`` gives
    $Z_{outer}=\zeta_c K_0(\gamma b)/(2\pi bK_1(\gamma b))$. Both shapes
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
    #: Cross-section shape of the inner conductor
    inner_shape: AbstractConductorShape = SchelkunoffRodShape()
    #: Cross-section shape of the shield, called with the shield's inner
    #: radius and its wall thickness, infinite when none is given
    shield_shape: AbstractConductorShape = SchelkunoffTubeShape()

    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor=None, shield_thickness=None) -> ImmittanceResult:
        return _coaxial_immittance(
            freq, d_in=d_in, d_out=d_out, dielectric=dielectric,
            conductor=conductor, outer_conductor=outer_conductor,
            shield_thickness=shield_thickness, inner_shape=self.inner_shape,
            shield_shape=self.shield_shape,
        )


class AbstractMicrostripFormulation(eqx.Module):
    """Abstract base class for a closed-form microstrip formulation."""
    @abstractmethod
    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        r"""
        Calculate the quasi-static solution.

        Parameters
        ----------
        w : ArrayLike
            Width of the microstrip trace in meters.
        h : ArrayLike
            Height of the dielectric substrate in meters.
        t : ArrayLike | None
            Trace thickness in meters, or ``None`` when unspecified.
        ep_r : jnp.ndarray
            Complex relative permittivity of the substrate, shape ``(npoints,)``.

        Returns
        -------
        PlanarQuasiStaticResult
            The effective permittivity, impedance and effective width.
        """
        raise NotImplementedError


class WheelerMicrostripFormulation(AbstractMicrostripFormulation):
    r"""
    Microstrip line formulation using the standard Wheeler approximations.

    This implements Wheeler's 1977 quasi-static approximation. It is distinct
    from the 1942 conductor-loss rule in
    :class:`~pmrf.models.components.lines.current_distribution.WheelerCurrentDistribution`
    and does not calculate conductor loss.

    **Mathematical Formulation**

    With ratio $u = W/H$, the effective relative permittivity ($\varepsilon_e$)
    and the impedance of the same geometry in air ($Z_a$) are:
    $$\varepsilon_e = \frac{\varepsilon_r + 1}{2} + \frac{\varepsilon_r - 1}{2} \frac{1}{\sqrt{1 + 12/u}}$$
    $$Z_a = \frac{120\pi}{u + 1.393 + 0.667 \ln(u + 1.444)} \quad (u > 1)
    \qquad
    Z_a = 60 \ln\left(\frac{8}{u} + \frac{u}{4}\right) \quad (u \leq 1)$$
    $$Z_c = \frac{Z_a}{\sqrt{\varepsilon_e}}$$

    Complex $\varepsilon_r$ carries dielectric loss through the same filling
    factor. The effective width is $W$.

    **Validity**

    Derived for a zero-thickness strip on an isotropic, non-magnetic substrate.
    Finite thickness is ignored by this formulation but remains available to
    the current distribution. Modal dispersion requires a separate
    :class:`AbstractMicrostripDispersion`.

    References
    ----------
    Wheeler, H. A. (1977). Transmission-Line Properties of a Strip on a Dielectric Sheet on a Plane.
    IEEE Transactions on Microwave Theory and Techniques.
    """
    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        # t is accepted and ignored: the 1977 result is derived for a
        # zero-thickness strip, so thickness enters neither ep_eff nor zc. It
        # still reaches the current distribution through the line's
        # cross-section, where it refines conductor loss.
        W, H = w, h
        u = W / H

        # Shared base terms
        t1 = (ep_r + 1) / 2
        t2 = (ep_r - 1) / 2
        t3 = 1 / jnp.sqrt(1 + 12 / u)

        # Piecewise effective permittivity (ep_eff)
        ep_eff_le1 = t1 + t2 * (t3 + 0.04 * (1 - u)**2)
        ep_eff_gt1 = t1 + t2 * t3
        ep_eff = jnp.where(u <= 1.0, ep_eff_le1, ep_eff_gt1) * jnp.ones_like(ep_r)

        # Piecewise characteristic impedance in air (Za)
        Za_le1 = 60 * jnp.log(8 / u + 0.25 * u)
        Za_gt1 = (120 * jnp.pi) / (u + 1.393 + 0.667 * jnp.log(u + 1.444))
        Za = jnp.where(u <= 1.0, Za_le1, Za_gt1)

        zc = Za / jnp.sqrt(ep_eff)
        w_eff = W * jnp.ones_like(ep_r)
        conductance_factor = _microstrip_conductance_factor(ep_r, ep_eff, zc)
        return PlanarQuasiStaticResult(ep_eff, zc, w_eff, conductance_factor)


class HammerstadJensenMicrostripFormulation(AbstractMicrostripFormulation):
    r"""
    Hammerstad--Jensen quasi-static microstrip formulation.

    **Mathematical Formulation**

    For normalized width $u=W/H$, the impedance of the homogeneous geometry is
    $$Z_L(u) = \frac{Z_0}{2\pi}\ln\left[\frac{F(u)}{u}
    + \sqrt{1 + \left(\frac{2}{u}\right)^2}\right]$$
    $$F(u) = 6 + (2\pi - 6)\exp\left[-(30.666/u)^{0.7528}\right].$$

    With normalized thickness $t_n=T/H$, finite conductor thickness changes the
    normalized widths in air and in the dielectric:
    $$\Delta u_1 = \frac{t_n}{\pi}\ln\left[1 + \frac{4e}{t_n}
    \tanh^2\left(\sqrt{6.517u}\right)\right]$$
    $$\Delta u_r = \frac{\Delta u_1}{2}\left[1 +
    \operatorname{sech}\left(\sqrt{\varepsilon_r-1}\right)\right],
    \qquad u_1=u+\Delta u_1,\quad u_r=u+\Delta u_r.$$

    The empirical exponents are
    $$a = 1 + \frac{1}{49}\ln\left[\frac{u_r^4+(u_r/52)^2}
    {u_r^4+0.432}\right] + \frac{1}{18.7}\ln\left[1+(u_r/18.1)^3\right]$$
    $$b = 0.564\left(\frac{\varepsilon_r-0.9}{\varepsilon_r+3}\right)^{0.053}.$$
    Defining
    $$e = \frac{\varepsilon_r + 1}{2}
    + \frac{\varepsilon_r - 1}{2}(1 + 10/u_r)^{-ab},$$
    the returned quantities are
    $$Z_c=\frac{Z_L(u_r)}{\sqrt{e}},\qquad
    \varepsilon_e=e\left[\frac{Z_L(u_1)}{Z_L(u_r)}\right]^2,
    \qquad W_{eff}=u_rH,$$
    $W_{eff}$ is passed to the dispersion formulation. Conductor loss is
    calculated separately by the current distribution.

    The fractional power in $b$ is evaluated through the principal logarithm.
    The dielectric constraint $\Re(\varepsilon_r)>1$ keeps its argument away
    from the negative-real branch cut for passive materials.

    **Validity**

    The quoted error in $\varepsilon_e$ is below 0.2% for
    $\varepsilon_r<128$ and $0.01\leq u\leq100$. The quoted error in $Z_L$
    is below 0.01% for $u\leq1$ and 0.03% for $u\leq1000$. The formulation
    is quasi-static and requires a non-magnetic substrate. Inputs outside the
    fitted ranges are extrapolated.

    References
    ----------
    Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
    Computer-Aided Design. IEEE MTT-S International Microwave Symposium Digest,
    407-409.
    """

    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        u = w / h
        thickness = None if t is None else t / h

        du1 = 0.0
        if thickness is not None:
            du1 = thickness / jnp.pi * jnp.log(
                1 + 4 * jnp.e / thickness * jnp.tanh(jnp.sqrt(6.517 * u)) ** 2
            )

        dur = du1 * (1 + 1 / jnp.cosh(jnp.sqrt(ep_r - 1))) / 2
        u1 = u + du1
        ur = u + dur

        zr = self._homogeneous_impedance(ur)
        z1 = self._homogeneous_impedance(u1)

        a = (
            1
            + jnp.log((ur**4 + (ur / 52) ** 2) / (ur**4 + 0.432)) / 49
            + jnp.log(1 + (ur / 18.1) ** 3) / 18.7
        )
        b_argument = (ep_r - 0.9) / (ep_r + 3)
        b = 0.564 * jnp.exp(0.053 * jnp.log(b_argument))
        e = (ep_r + 1) / 2 + (ep_r - 1) / 2 * (1 + 10 / ur) ** (-a * b)

        zc = zr / jnp.sqrt(e)
        ep_eff = e * (z1 / zr) ** 2
        w_eff = ur * h
        conductance_factor = _microstrip_conductance_factor(ep_r, ep_eff, zc)
        return PlanarQuasiStaticResult(ep_eff, zc, w_eff, conductance_factor)

    @staticmethod
    def _homogeneous_impedance(u):
        z0 = jnp.sqrt(mu_0 / epsilon_0)
        f_u = 6 + (2 * jnp.pi - 6) * jnp.exp(-(30.666 / u) ** 0.7528)
        return z0 / (2 * jnp.pi) * jnp.log(f_u / u + jnp.sqrt(1 + (2 / u) ** 2))


class AbstractMicrostripDispersion(eqx.Module):
    """Abstract modal-dispersion formulation for an inhomogeneous microstrip."""

    @abstractmethod
    def disperse(
        self, freq: Frequency, *, ep_eff_0, zc_0, ep_r, w_eff, h
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Return frequency-dependent $(\varepsilon_e, Z_c)$.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        ep_eff_0 : jnp.ndarray
            Quasi-static complex effective relative permittivity.
        zc_0 : jnp.ndarray
            Quasi-static characteristic impedance in ohms.
        ep_r : jnp.ndarray
            Complex relative permittivity of the substrate.
        w_eff : jnp.ndarray
            Electromagnetic effective conductor width in meters.
        h : ArrayLike
            Substrate height in meters.

        Returns
        -------
        tuple of jnp.ndarray
            The dispersed effective permittivity and characteristic impedance.
        """
        raise NotImplementedError


class KirschningJansenMicrostripDispersion(AbstractMicrostripDispersion):
    r"""
    Kirschning--Jansen modal dispersion for microstrip.

    **Mathematical Formulation**

    The model expresses the dispersed effective permittivity as
    $$\varepsilon_e(f) = \varepsilon_r
    - \frac{\varepsilon_r-\varepsilon_e(0)}{1 + P(f)},$$
    where $P=P_1P_2(0.1844+P_3P_4)^{1.5763}f_n^{1.5763}$ and the $P_i$
    are
    $$P_1=0.27488+\left[0.6315+\frac{0.525}{(1+0.0157f_n)^{20}}\right]u
    -0.065683e^{-8.7513u}$$
    $$P_2=0.33622(1-e^{-0.03442\varepsilon_r}),\quad
    P_3=0.0363e^{-4.6u}[1-e^{-(f_n/38.7)^{4.97}}]$$
    $$P_4=1+2.751[1-e^{-(\varepsilon_r/15.916)^8}].$$
    The normalized frequency is
    $f_n=f[\mathrm{Hz}]H[\mathrm{m}]10^{-6}$ (GHz-mm).
    The normalized width is the thickness-corrected $u = W_{eff}/H$.

    Characteristic impedance is corrected by
    $$Z_c(f)=Z_c(0)\left(\frac{R_{13}}{R_{14}}\right)^{R_{17}},$$
    with
    $$R_1=\min(0.03891\varepsilon_r^{1.4},20),\quad
    R_2=\min(0.2671u^7,20),\quad R_3=4.766e^{-3.228u^{0.641}}$$
    $$R_4=0.016+(0.0514\varepsilon_r)^{4.524},\quad
    R_5=(f_n/28.843)^{12},\quad R_6=\min(22.20u^{1.92},20)$$
    $$R_7=1.206-0.3144e^{-R_1}(1-e^{-R_2})$$
    $$R_8=1+1.275\left[1-e^{-0.004625R_3\varepsilon_r^{1.674}
    (f_n/18.365)^{2.745}}\right]$$
    $$R_9=\frac{5.086R_4R_5e^{-R_6}}{(0.3838+0.386R_4)(1+1.2992R_5)}
    \frac{(\varepsilon_r-1)^6}{1+10(\varepsilon_r-1)^6}$$
    $$R_{10}=0.00044\varepsilon_r^{2.136}+0.0184,\quad
    R_{11}=\frac{(f_n/19.47)^6}{1+0.0962(f_n/19.47)^6},\quad
    R_{12}=\frac{1}{1+0.00245u^2}$$
    $$R_{13}=0.9408\varepsilon_e(f)^{R_8}-0.9603,\quad
    R_{14}=(0.9408-R_9)\varepsilon_e(0)^{R_8}-0.9603$$
    $$R_{15}=0.707R_{10}(f_n/12.3)^{1.097},\quad
    R_{16}=1+0.0503\varepsilon_r^2R_{11}[1-e^{-(u/15)^6}]$$
    $$R_{17}=R_7\left[1-\frac{1.1241R_{12}}{R_{16}}
    e^{-0.026f_n^{1.15656}-R_{15}}\right].$$

    The papers validate the fit from $\varepsilon_r=2.2$, while modal
    dispersion must vanish at the homogeneous $\varepsilon_r=1$ limit. In the
    extrapolation interval ParamRF therefore applies the smooth weight
    $$x=\operatorname{clip}\left(\frac{\Re(\varepsilon_r)-1}{1.2},0,1\right),
    \qquad q=x^2(3-2x),$$
    $$\varepsilon_e=(1-q)\varepsilon_e(0)+q\varepsilon_{e,KJ},\qquad
    Z_c=(1-q)Z_c(0)+qZ_{c,KJ}.$$
    This smooth extension to the homogeneous limit is specific to ParamRF.

    **Validity**

    The published fits cover $1\leq\varepsilon_r\leq20$,
    $0.1\leq W/H\leq100$, and $0\leq H/\lambda_0\leq0.13$, with numerical
    fits anchored at $\varepsilon_r\geq2.2$. ParamRF uses the extension above
    for $1<\varepsilon_r<2.2$ and extrapolates outside the remaining ranges.

    References
    ----------
    Kirschning, M., & Jansen, R. H. (1982). Accurate Model for Effective
    Dielectric Constant of Microstrip with Validity up to Millimeter-Wave
    Frequencies. Electronics Letters, 18(6), 272-273.

    Jansen, R. H., & Kirschning, M. (1983). Arguments and an Accurate Model for
    the Power-Current Formulation of Microstrip Characteristic Impedance.
    Archiv fuer Elektronik und Uebertragungstechnik, 37, 108-112.
    """

    def disperse(
        self, freq: Frequency, *, ep_eff_0, zc_0, ep_r, w_eff, h
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        u = w_eff / h
        fn = freq.f * h * 1e-6

        p1 = (
            0.27488
            + (0.6315 + 0.525 / (1 + 0.0157 * fn) ** 20) * u
            - 0.065683 * jnp.exp(-8.7513 * u)
        )
        p2 = 0.33622 * (1 - jnp.exp(-0.03442 * ep_r))
        p3 = 0.0363 * jnp.exp(-4.6 * u) * (1 - jnp.exp(-(fn / 38.7) ** 4.97))
        p4 = 1 + 2.751 * (1 - jnp.exp(-(ep_r / 15.916) ** 8))
        p_f = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
        ep_eff = ep_r - (ep_r - ep_eff_0) / (1 + p_f)

        r1_raw = 0.03891 * ep_r**1.4
        r1 = jnp.where(jnp.real(r1_raw) < 20, r1_raw, 20)
        r2 = jnp.minimum(0.2671 * u**7, 20)
        r3 = 4.766 * jnp.exp(-3.228 * u**0.641)
        r4 = 0.016 + (0.0514 * ep_r) ** 4.524
        r5 = (fn / 28.843) ** 12
        r6 = jnp.minimum(22.20 * u**1.92, 20)
        r7 = 1.206 - 0.3144 * jnp.exp(-r1) * (1 - jnp.exp(-r2))
        r8 = 1 + 1.275 * (
            1 - jnp.exp(-0.004625 * r3 * ep_r**1.674 * (fn / 18.365) ** 2.745)
        )
        r9 = (
            5.086
            * r4
            * r5
            / (0.3838 + 0.386 * r4)
            * jnp.exp(-r6)
            / (1 + 1.2992 * r5)
            * (ep_r - 1) ** 6
            / (1 + 10 * (ep_r - 1) ** 6)
        )
        r10 = 0.00044 * ep_r**2.136 + 0.0184
        r11 = (fn / 19.47) ** 6 / (1 + 0.0962 * (fn / 19.47) ** 6)
        r12 = 1 / (1 + 0.00245 * u**2)
        r13 = 0.9408 * ep_eff**r8 - 0.9603
        r14 = (0.9408 - r9) * ep_eff_0**r8 - 0.9603
        r15 = 0.707 * r10 * (fn / 12.3) ** 1.097
        r16 = 1 + 0.0503 * ep_r**2 * r11 * (1 - jnp.exp(-(u / 15) ** 6))
        r17 = r7 * (
            1
            - 1.1241
            * r12
            / r16
            * jnp.exp(-0.026 * fn**1.15656 - r15)
        )
        zc = zc_0 * (r13 / r14) ** r17

        # The published fit starts at epsilon_r=2.2, while modal dispersion must
        # vanish at the homogeneous epsilon_r=1 limit. Smoothly introduce the
        # empirical correction over that extrapolation interval. The smoothstep
        # has zero slope at both ends, keeping values and gradients continuous.
        fit_fraction = jnp.clip((jnp.real(ep_r) - 1) / 1.2, 0, 1)
        modal_weight = fit_fraction**2 * (3 - 2 * fit_fraction)
        ep_eff = ep_eff_0 + modal_weight * (ep_eff - ep_eff_0)
        zc = zc_0 + modal_weight * (zc - zc_0)
        return ep_eff, zc


class AbstractStriplineFormulation(eqx.Module):
    """Abstract base class for a closed-form stripline formulation.

    Homogeneous filling gives $\varepsilon_e=\varepsilon_r$ without modal
    dispersion.
    """

    @abstractmethod
    def quasi_static(self, *, w, b, t, ep_r) -> PlanarQuasiStaticResult:
        r"""
        Calculate the quasi-static solution.

        Parameters
        ----------
        w : ArrayLike
            Width of the centre strip in meters.
        b : ArrayLike
            Ground-plane separation in meters.
        t : ArrayLike | None
            Thickness of the strip in meters, or None for a zero-thickness strip.
        ep_r : jnp.ndarray
            Complex relative permittivity of the filling, shape ``(npoints,)``.

        Returns
        -------
        PlanarQuasiStaticResult
            The effective permittivity, impedance and effective width.
        """
        raise NotImplementedError


class CohnStriplineFormulation(AbstractStriplineFormulation):
    r"""
    Cohn's stripline formulation, in the form tabulated by Pozar.

    **Mathematical Formulation**

    The filling is homogeneous, so
    $$\varepsilon_e = \varepsilon_r$$
    exactly, with no filling factor and no modal dispersion. With the fringing
    correction to the strip width,
    $$\frac{W_e}{b} = \frac{W}{b} -
    \begin{cases}0, & W/b > 0.35,\\ (0.35 - W/b)^2, & W/b \leq 0.35,\end{cases}$$
    the characteristic impedance of the zero-thickness strip is
    $$Z_c = \frac{30\pi}{\sqrt{\varepsilon_r}}\frac{b}{W_e + 0.441b}.$$

    Conductor loss is supplied by
    :class:`~pmrf.models.components.lines.current_distribution.CohnCurrentDistribution`,
    and complex $\varepsilon_e$ carries dielectric loss.

    **Validity**

    The impedance expression assumes zero thickness and uses a continuous
    fringing correction at $W/b=0.35$. A supplied thickness must satisfy
    $0<T<b$, although it does not enter the impedance expression.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE Transactions
    on Microwave Theory and Techniques, 3(2), 119-126.

    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.7. Wiley.
    """

    def quasi_static(self, *, w, b, t, ep_r) -> PlanarQuasiStaticResult:
        ones = jnp.ones_like(ep_r)
        if t is not None:
            t = eqx.error_if(t, t - b >= 0, "stripline thickness must satisfy 0 < t < b")
        ep_eff = ep_r * ones

        u = w / b
        w_e = b * (u - jnp.where(u > 0.35, 0.0, (0.35 - u) ** 2))
        zc = 30 * jnp.pi / jnp.sqrt(ep_eff) * b / (w_e + 0.441 * b)

        shunt_conductance_factor = jnp.sqrt(ep_eff) / (zc * c * epsilon_0 * ep_eff)
        return PlanarQuasiStaticResult(
            ep_eff, zc, w_e * ones, shunt_conductance_factor,
        )

def _microstrip_conductance_factor(ep_r, ep_eff, zc):
    """Convert static substrate conductivity through the quasi-static fill."""
    capacitance = jnp.sqrt(ep_eff) / (zc * c)
    delta = ep_r - 1
    safe_delta = jnp.where(jnp.abs(delta) > 1e-14, delta, 1.0)
    filling = jnp.where(
        jnp.abs(delta) > 1e-14,
        (capacitance - 1 / (zc * c * jnp.sqrt(ep_eff))) / safe_delta,
        0.0,
    )
    return jnp.real(filling) / epsilon_0
