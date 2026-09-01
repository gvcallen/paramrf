"""
Closed-form physics for transmission lines (coaxial, microstrip, stripline).

Three distinct strategy roles appear on a line model, each with its own field:

- A **Formulation** (`formulation`) produces the complete electrical state a
  model needs to reach S-parameters: either a per-unit-length immittance
  directly, or a :class:`PlanarQuasiStaticResult` the line converts into one.
  It is the primary strategy, and a line always has exactly one.
- A **Dispersion** (`dispersion`) modifies an existing quasi-static state with
  modal frequency dependence. It never produces a state of its own, so it only
  exists where the cross-section is inhomogeneous and the mode is therefore not
  strictly TEM -- microstrip has one, homogeneously filled coaxial and stripline
  do not.
- A **Roughness** (`roughness`, in :mod:`pmrf.materials.conductor`) modifies
  conductor behaviour rather than line state: it scales a surface impedance,
  and so belongs to the conductor material, not to the geometry.

A formulation is pure numerics. Materials are evaluated by the line, so a
formulation never sees a :class:`~pmrf.Param` or a :class:`~pmrf.Module`
parameter: it can be checked directly against the equations of the paper it
comes from, and contributed without learning the material taxonomy.
"""
from abc import abstractmethod
from scipy.constants import c, mu_0, epsilon_0
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.materials import ConductorProperties, DielectricProperties
from pmrf.materials.conductor_shape import (
    AbstractConductorShape,
    HalfSpaceShape,
    SchelkunoffRodShape,
    SchelkunoffTubeShape,
    SchelkunoffCothTubeShape,
    SchelkunoffInfiniteTubeShape,
    TescheRodShape,
    TescheTubeShape,
)
from pmrf.models.components.lines.current_distribution import (
    AbstractCurrentDistribution,
    CohnCurrentDistribution,
    WheelerCurrentDistribution,
    _wheeler_current_factor,
)
from pmrf.models.components.lines.base import ImmittanceResult


class PlanarQuasiStaticResult(eqx.Module):
    r"""
    Quasi-static solution of a single-conductor planar line over a ground plane.

    A quasi-static formulation stops at the effective permittivity, the
    characteristic impedance it implies and the effective conductor width; the
    line turns those into an immittance, and a dispersion model may correct them
    first.

    Coupled lines are not this type: they carry a solution per mode, so they
    need their own result.

    Parameters
    ----------
    ep_eff : jnp.ndarray
        Complex effective relative permittivity, shape ``(npoints,)``.
    zc : jnp.ndarray
        Quasi-static characteristic impedance in ohms, $Z_a/\sqrt{\varepsilon_e}$.
    w_eff : jnp.ndarray
        Electromagnetic effective conductor width in meters.
    conductor_loss_factor : jnp.ndarray | None
        Legacy scalar geometry factor retained for direct callers. Line models
        use a current-distribution strategy instead.
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

    #: Legacy geometry factor for callers predating current distributions
    conductor_loss_factor: jnp.ndarray | None = None

    def to_immittance(
        self, freq: Frequency, dielectric: DielectricProperties,
        conductor: ConductorProperties,
        current_distribution: AbstractCurrentDistribution | None = None,
        **geometry,
    ) -> ImmittanceResult:
        r"""
        Converts the quasi-static solution into a per-unit-length immittance.

        The external inductance and the shunt admittance follow from the
        quasi-static impedance and effective permittivity. Surface impedance
        is charged through the supplied current-distribution strategy:
        $$Z = \frac{j\omega Z_c \sqrt{\varepsilon_e}}{c} + Z_s K_c
        \qquad
        Y = \frac{j\omega \sqrt{\varepsilon_e}}{Z_c c}$$

        Each distribution returns one or more conductor shapes and their
        inverse-metre geometry weights. This keeps conductor-loss selection
        independent from the quasi-static formulation.

        $\varepsilon_e$ is complex, so $Y$ already carries the dielectric loss
        as its real part and needs no separate loss-tangent term.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        zs : jnp.ndarray
            Complex surface impedance of the conductor in ohm per square.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance.

        References
        ----------
        Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.8. Wiley.
        """
        w = freq.w
        sqrt_ep_eff_mu = jnp.sqrt(self.ep_eff * dielectric.mu_r)
        sqrt_ep_eff_over_mu = jnp.sqrt(self.ep_eff / dielectric.mu_r)

        Z = 1j * w * self.zc * sqrt_ep_eff_mu / c
        if current_distribution is None:
            if self.conductor_loss_factor is None:
                raise ValueError("current_distribution is required")
            Z_cond = conductor.zs * self.conductor_loss_factor
        else:
            Z_cond = sum(
                shape.impedance(w, conductor) * weight
                for shape, weight in current_distribution.distribute(
                    freq, zc=self.zc, conductor=conductor, **geometry
                )
            )
        Z = Z + Z_cond
        Y = 1j * w * sqrt_ep_eff_over_mu / (self.zc * c)
        Y = Y + dielectric.sigma * self.shunt_conductance_factor

        return ImmittanceResult(Z=Z, Y=Y, w=w)


class AbstractCoaxialFormulation(eqx.Module):
    """
    Abstract base class for a closed-form coaxial line formulation.

    A formulation is pure numerics: all material arguments arrive as evaluated
    arrays in small property records, so it can be exercised directly against
    its published equations with no ParamRF objects in sight.
    """
    @abstractmethod
    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor: ConductorProperties | None = None, shield_thickness=None) -> ImmittanceResult:
        r"""
        Calculates the per-unit-length immittance of the line.

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
    """Assemble a coaxial line's immittance around a choice of inner-conductor shape.

    Everything but the inner rod is common to every coaxial formulation here:
    the external inductance and the shunt admittance are pure geometry, and
    the shield's contribution is selected from the cylindrical tube shapes. The formulations
    differ only in how the solid centre conductor's cross-section is solved.
    """
    eps = epsilon_0 * dielectric.ep_r
    mu = mu_0 * dielectric.mu_r
    w = freq.w

    d_out = eqx.error_if(d_out, d_out - d_in <= 0, "d_out must exceed d_in")
    a, b = d_in / 2, d_out / 2
    ln_b_over_a = jnp.log(b / a)

    L_ext = jnp.ones(freq.npoints) * mu / (2 * jnp.pi) * ln_b_over_a

    rod_zs = inner_shape.impedance(w, conductor, a=a)
    Z_int = rod_zs / (2 * jnp.pi * a)
    # An unspecified wall is the infinite-wall cylindrical limit. A finite wall
    # uses Schelkunoff's tube solution; its cheaper coth-plus-curvature sibling
    # remains available as a public cross-section entry for large sweeps.
    shield_conductor = conductor if outer_conductor is None else outer_conductor
    if shield_thickness is None:
        if isinstance(shield_shape, HalfSpaceShape):
            shield_zs = shield_shape.impedance(w, shield_conductor)
        else:
            shield_zs = shield_shape.impedance(w, shield_conductor, a=b)
    else:
        shield_zs = shield_shape.impedance(
            w, shield_conductor, a=b, t=shield_thickness
        )
    Z_int = Z_int + shield_zs / (2 * jnp.pi * b)

    Z = 1j * w * L_ext + Z_int
    Y = 1j * w * 2 * jnp.pi * eps / ln_b_over_a
    Y = Y + 2 * jnp.pi * dielectric.sigma / ln_b_over_a

    return ImmittanceResult(Z=Z, Y=Y, w=w)


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

    which is $G + j\omega C$ with $C = 2\pi\Re(\varepsilon)/\ln(b/a)$ and
    $G = -2\pi\omega\Im(\varepsilon)/\ln(b/a)$. Each conductor's series
    impedance is a shape's surface impedance charged over its circumference,
    $Z = Z_s/2\pi r$: the inner conductor is a
    :class:`~pmrf.materials.conductor_shape.TescheRodShape`, and the outer
    shield -- treated as infinitely thick because only its inner diameter is
    part of :class:`CoaxialLine` -- is the
    :class:`~pmrf.materials.conductor_shape.HalfSpaceShape` limit that Tesche's
    tube circuit approaches as its wall thickens; the finite-wall form is
    :class:`~pmrf.materials.conductor_shape.TescheTubeShape`.

    A magnetic filling needs no special case. With complex $\mu_r$ the external
    term $j\omega L'$ acquires the real part $\omega\mu''\ln(b/a)/2\pi$, so
    magnetic loss enters the series resistance on its own.

    **Validity**

    The equivalent circuit is not an empirical fit to a width ratio, so it
    carries no geometric fit range: it interpolates between the exact DC
    resistance and the bare half-space impedance $\zeta_c$. That is not the
    exact high-frequency limit of a round conductor -- the circuit reaches
    $R_{dc} + Z_{hf}$ and never captures the $1/2\gamma a$ curvature term of
    the exact solution -- so it is an approximation to
    :class:`SchelkunoffCoaxialFormulation` at every finite frequency, worst
    for thin inner conductors at low frequency. The transmission-line
    description itself is the further limit: it assumes
    the TEM mode, so it holds below the TE11 cutoff
    $f_c \approx c / \left[\pi (a + b) \sqrt{\varepsilon_r \mu_r}\right]$.
    Above that, higher-order modes propagate and a single per-unit-length
    immittance no longer describes the line.

    References
    ----------
    Tesche, F. M. (2007). A Simple Model for the Line Parameters of a Lossy Coaxial
    Cable Filled With a Nondispersive Dielectric. IEEE Transactions on Electromagnetic
    Compatibility, 49(1), 12-17.

    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial Transmission Lines
    and Cylindrical Shields. Bell System Technical Journal, 13(4), 532-579.
    """
    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor=None, shield_thickness=None) -> ImmittanceResult:
        return _coaxial_immittance(
            freq, d_in=d_in, d_out=d_out, dielectric=dielectric,
            conductor=conductor, outer_conductor=outer_conductor,
            shield_thickness=shield_thickness, inner_shape=TescheRodShape(),
            shield_shape=(HalfSpaceShape() if shield_thickness is None else TescheTubeShape()),
        )


class SchelkunoffCoaxialFormulation(AbstractCoaxialFormulation):
    r"""
    Coaxial line formulation using Schelkunoff's exact cylindrical solution.

    **Mathematical Formulation**

    The external inductance and shunt admittance are the same pure geometry
    as in :class:`TescheCoaxialFormulation`,
    $$L' = \frac{\mu_0 \mu_r}{2\pi} \ln\left(\frac{b}{a}\right)
    \qquad
    Y = \frac{j\omega 2\pi\varepsilon}{\ln(b/a)},$$
    The inner conductor uses
    :class:`~pmrf.materials.conductor_shape.SchelkunoffRodShape` --
    Schelkunoff's eq. (65) rather than an equivalent circuit:
    $$Z_{inner} = \frac{\zeta_c}{2\pi a}\,
    \frac{I_0(\gamma a)}{I_1(\gamma a)},\qquad
    \gamma = \sqrt{j\omega\mu\sigma}.$$

    An unspecified shield thickness uses the matching infinite-wall tube
    solution, $Z_{outer}=\zeta_c K_0(\gamma b)/(2\pi bK_1(\gamma b))$;
    a specified thickness uses Schelkunoff's finite-tube equation.

    **Validity**

    Exact for the inner conductor at every frequency, dc included, so unlike
    Tesche it leaves no frequency-shaped residual for a conductivity fit to
    absorb. The shield remains infinitely thick, since only its inner
    diameter is part of :class:`CoaxialLine`. The transmission-line
    description is still the outer limit: it assumes the TEM mode, so it
    holds below the TE11 cutoff
    $f_c \approx c / \left[\pi (a + b) \sqrt{\varepsilon_r \mu_r}\right]$.

    References
    ----------
    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial Transmission Lines
    and Cylindrical Shields. Bell System Technical Journal, 13(4), 532-579. Eq. (65).
    """
    def immittance(self, freq: Frequency, *, d_in, d_out, dielectric: DielectricProperties, conductor: ConductorProperties, outer_conductor=None, shield_thickness=None) -> ImmittanceResult:
        return _coaxial_immittance(
            freq, d_in=d_in, d_out=d_out, dielectric=dielectric,
            conductor=conductor, outer_conductor=outer_conductor,
            shield_thickness=shield_thickness, inner_shape=SchelkunoffRodShape(),
            shield_shape=(SchelkunoffInfiniteTubeShape() if shield_thickness is None else SchelkunoffTubeShape()),
        )


class AbstractMicrostripFormulation(eqx.Module):
    """
    Abstract base class for a closed-form microstrip line formulation.

    A formulation is pure numerics: every argument arrives as an
    already-evaluated array, so it can be exercised directly against its
    published equations with no ParamRF objects in sight.
    """
    @abstractmethod
    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        r"""
        Calculates the quasi-static solution of the line.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        w : ArrayLike
            Width of the microstrip trace in meters.
        h : ArrayLike
            Height of the dielectric substrate in meters.
        t : ArrayLike | None
            Thickness of the trace in meters, or None for a zero-thickness trace.
        ep_r : jnp.ndarray
            Complex relative permittivity of the substrate, shape ``(npoints,)``.
        zs : jnp.ndarray
            Complex surface impedance of the conductor in ohm per square,
            shape ``(npoints,)``.

        Returns
        -------
        PlanarQuasiStaticResult
            The effective permittivity, impedance and effective width.
        """
        raise NotImplementedError


class WheelerMicrostripFormulation(AbstractMicrostripFormulation):
    r"""
    Microstrip line formulation using the standard Wheeler approximations.

    **Mathematical Formulation**

    With ratio $u = W/H$, the effective relative permittivity ($\varepsilon_e$)
    and the impedance of the same geometry in air ($Z_a$) are:
    $$\varepsilon_e = \frac{\varepsilon_r + 1}{2} + \frac{\varepsilon_r - 1}{2} \frac{1}{\sqrt{1 + 12/u}}$$
    $$Z_a = \frac{120\pi}{u + 1.393 + 0.667 \ln(u + 1.444)} \quad (u > 1)
    \qquad
    Z_a = 60 \ln\left(\frac{8}{u} + \frac{u}{4}\right) \quad (u \leq 1)$$
    $$Z_c = \frac{Z_a}{\sqrt{\varepsilon_e}}$$

    $\varepsilon_r$ is complex, and $\varepsilon_e$ is linear in it, so the
    dielectric loss carries through the same filling factor as the real part and
    needs no separate loss-tangent term. The effective width is $W$: the
    approximation is derived for a zero-thickness strip, so `zs` does not enter
    here. Conductor loss is charged separately, through
    `conductor_loss_factor`, by Wheeler's own incremental-inductance rule
    (see :func:`_wheeler_conductor_loss_factor`), applied over the physical
    width $W$ rather than a thickness-widened one.

    **Validity**

    Derived for a zero-thickness strip on an isotropic, non-magnetic substrate,
    which is why finite thickness is rejected rather than ignored. It is a
    quasi-static result and carries no modal dispersion, so it describes the
    line only well below the frequency at which $\varepsilon_e$ begins to rise
    towards $\varepsilon_r$; pair it with an
    :class:`AbstractMicrostripDispersion` above that. Outside its fitted width
    range the closed form remains smooth and finite, so ParamRF extrapolates
    rather than rejecting.

    References
    ----------
    Wheeler, H. A. (1977). Transmission-Line Properties of a Strip on a Dielectric Sheet on a Plane.
    IEEE Transactions on Microwave Theory and Techniques.
    """
    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        if t is not None:
            raise ValueError("Wheeler microstrip approximation does not support finite thickness")

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
        return PlanarQuasiStaticResult(
            ep_eff, zc, w_eff, conductance_factor,
            _wheeler_conductor_loss_factor(W, zc),
        )


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
    where $e$ denotes the bracketed permittivity expression before its
    thickness correction. $W_{eff}$ is the electromagnetic fringing width and
    feeds the dispersion formulation; conductor loss is charged separately,
    through `conductor_loss_factor`, by Wheeler's incremental-inductance rule
    (see :func:`_wheeler_conductor_loss_factor`) over the physical width $W$,
    which is the correct one for that rule regardless of which formulation
    supplied the quasi-static solution.

    The fractional power in $b$ is evaluated through the principal logarithm.
    The dielectric constraint $\Re(\varepsilon_r)>1$ keeps its argument away
    from the negative-real branch cut for passive materials.

    **Validity**

    Hammerstad and Jensen quote the fit accuracy of $\varepsilon_e$ as better
    than 0.2% for $\varepsilon_r < 128$ and $0.01 \leq u \leq 100$, and the
    accuracy of $Z_L$ as better than 0.01% for $u \leq 1$ and 0.03% for
    $u \leq 1000$. The result is quasi-static: it carries no modal dispersion,
    so a :class:`AbstractMicrostripDispersion` supplies the frequency
    dependence. The substrate must be non-magnetic; :class:`MicrostripLine`
    rejects $\mu_r \neq 1$ rather than extrapolating. Leaving the fitted ranges
    is a documented extrapolation: the expressions stay finite and smooth, and
    only $\Re(\varepsilon_r) > 1$ is enforced, because the fractional power in
    $b$ has a branch cut below it.

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
        return PlanarQuasiStaticResult(
            ep_eff, zc, w_eff, conductance_factor,
            _wheeler_conductor_loss_factor(w, zc),
        )

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
    This extension is not part of the cited empirical fit; it supplies a
    continuous, differentiable connection to its required homogeneous limit.

    **Validity**

    The published fits cover $1 \leq \varepsilon_r \leq 20$,
    $0.1 \leq W/H \leq 100$ and $0 \leq H/\lambda_0 \leq 0.13$, with the
    numerical fits themselves anchored from $\varepsilon_r = 2.2$ upwards; the
    smooth homogeneous-limit weight described above covers the interval below
    that, and is ParamRF's own extension rather than part of the cited fit.
    Outside these ranges the expressions remain finite, so ParamRF extrapolates
    and documents it rather than rejecting the input.

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
    """
    Abstract base class for a closed-form stripline formulation.

    Stripline is homogeneously filled, so there is no modal dispersion to
    correct afterwards: the quasi-static solution is the solution, and its
    effective permittivity is the substrate permittivity itself.

    A formulation is pure numerics: every argument arrives as an
    already-evaluated array, so it can be exercised directly against its
    published equations with no ParamRF objects in sight.
    """

    @abstractmethod
    def quasi_static(self, *, w, b, t, ep_r) -> PlanarQuasiStaticResult:
        r"""
        Calculates the quasi-static solution of the line.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        w : ArrayLike
            Width of the centre strip in meters.
        b : ArrayLike
            Ground-plane separation in meters.
        t : ArrayLike | None
            Thickness of the strip in meters, or None for a zero-thickness strip.
        ep_r : jnp.ndarray
            Complex relative permittivity of the filling, shape ``(npoints,)``.
        zs : jnp.ndarray
            Complex surface impedance of the conductor in ohm per square,
            shape ``(npoints,)``.

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

    Conductor loss follows Cohn's incremental-inductance result, which unlike
    the impedance does depend on the strip thickness $T$:
    $$\frac{\alpha_c}{R_s} = \begin{cases}
    \dfrac{2.7\times10^{-3}\,\varepsilon_r Z_c}{30\pi(b-T)}A,
        & \sqrt{\varepsilon_r}Z_c < 120,\\[2ex]
    \dfrac{0.16}{Z_c b}B, & \sqrt{\varepsilon_r}Z_c \geq 120,
    \end{cases}$$
    $$A = 1 + \frac{2W}{b-T}
    + \frac{1}{\pi}\frac{b+T}{b-T}\ln\frac{2b-T}{T}$$
    $$B = 1 + \frac{b}{0.5W + 0.7T}
    \left(0.5 + \frac{0.7T}{W} + \frac{1}{2\pi}\ln\frac{4\pi W}{T}\right).$$

    That attenuation is returned directly as the conductor-loss factor, not
    disguised as a width. A per-unit-length series resistance is
    $R = 2\alpha_c Z_c$, so the factor multiplying the surface impedance is
    $$K_c = 2\frac{\alpha_c}{R_s}Z_c,$$
    which is independent of $R_s$. The returned $W_e$ stays the genuine
    electromagnetic fringing width and is not reused for loss. Dielectric loss
    needs no separate term: $\varepsilon_e$ is complex, and carries it.

    The formulas are the standard piecewise fits, discontinuous by a fraction of
    a percent at $\sqrt{\varepsilon_r}Z_c = 120$; the branch is selected on the
    real parts.

    **Validity**

    The filling is homogeneous, so $\varepsilon_e = \varepsilon_r$ is exact and
    there is no fit range on the permittivity and no modal dispersion below the
    first higher-order mode. The impedance expression is Cohn's zero-thickness
    result with a fringing correction whose two branches meet at $W/b = 0.35$;
    the attenuation fit is piecewise in $\sqrt{\varepsilon_r} Z_c$ and is
    discontinuous there by a fraction of a percent. Finite thickness must
    satisfy $0 < T < b$, which is enforced, and the attenuation diverges
    logarithmically as $T \to 0$, so a zero-thickness strip is given no
    conductor loss rather than an infinite one.

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

def _wheeler_conductor_loss_factor(w, zc):
    r"""Wheeler's incremental-inductance rule, as a `conductor_loss_factor`.

    $$k_c = \frac{2}{W}\exp\left[-1.2\left(\frac{\Re(Z_c)}{Z_0}\right)^{0.7}\right]$$

    charges the sheet impedance over the physical trace width $W$, not the
    dispersion-widened $W_{eff}$: the rule sums over every receded conductor
    surface, and those terms do not vanish as $t\to0$, so the effective width
    used to widen the current-carrying geometry is not the right one here.
    $Z_c$ is the (possibly dispersed) characteristic impedance at the point in
    the line's pipeline where the factor is evaluated; only its real part
    enters, since the exponent is a lossless current-crowding correction.

    This is the low-loss linearisation shared by both callers: charged
    directly onto $\gamma$ it reproduces the rule's own $\alpha_c$, and
    charged onto $Z$ through :meth:`PlanarQuasiStaticResult.to_immittance` it
    reduces to the same $\alpha_c$ once $Z=\gamma Z_c$ is inverted, using
    $\Re(\gamma) \approx \Re(Z_s k_c)/(2\Re(Z_c))$ for a small series
    perturbation on an otherwise lossless line.

    References
    ----------
    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.
    """
    return 2 / w * _wheeler_current_factor(zc)


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
