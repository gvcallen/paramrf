"""Stripline models, formulations, and current distributions."""
from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.materials import AbstractConductor, AbstractDielectric, BulkConductor, ConstantDielectric, as_conductor, as_dielectric
from pmrf.materials.surface_impedance import AbstractSurfaceImpedance, EvenOddSlabSurfaceImpedance, HalfSpaceSurfaceImpedance
from pmrf.models.components.lines.base import AbstractImmittanceLine, ImmittanceResult
from pmrf.models.components.lines.planar import AbstractCurrentDistribution, AbstractPlanarCrossSection, PlanarQuasiStaticResult
from pmrf.parameters import Param, as_param, param
from pmrf.utils import field

class StriplineCrossSection(AbstractPlanarCrossSection):
    """Cross-section of a centre strip between two ground planes.

    Parameters
    ----------
    w : ArrayLike
        Width of the centre strip in meters.
    b : ArrayLike
        Separation of the ground planes in meters.
    t : ArrayLike | None, default=None
        Strip thickness in meters, or ``None`` when it is unspecified.
    ep_r : jnp.ndarray | None, default=None
        Complex relative permittivity of the homogeneous filling, used by
        Cohn's attenuation model.
    """

    #: Width of the centre strip in meters
    w: jnp.ndarray

    #: Separation of the ground planes in meters
    b: jnp.ndarray

    #: Strip thickness in meters, or ``None`` when unspecified
    t: jnp.ndarray | None = None

    #: Complex relative permittivity of the filling
    ep_r: jnp.ndarray | None = None

    def dimensions(self) -> dict:
        return {"w": self.w, "t": self.t}


class CohnCurrentDistribution(AbstractCurrentDistribution[StriplineCrossSection]):
    r"""Cohn's stripline current distribution.

    **Mathematical Formulation**

    Cohn gives conductor attenuation per unit length. Inverting
    $$\alpha_c=\frac{\Re(Z_s k_c)}{2\Re(Z_c)}$$
    gives the geometry weight
    $$k_c=2(\alpha_c/R_s)\Re(Z_c),$$
    with $\alpha_c/R_s$ taken from Cohn's wide-strip expression when
    $\sqrt{\varepsilon_r}\,\Re(Z_c)<120$ and from his narrow-strip expression
    otherwise, as tabulated by Pozar. $k_c$ is frequency-independent, and it
    covers the centre strip's two faces, its edges, and both ground planes
    together -- Cohn's $\alpha_c$ is the whole line's conductor attenuation.

    The weight is paired with :attr:`slab_impedance`, by default
    :class:`~pmrf.materials.surface_impedance.EvenOddSlabSurfaceImpedance`,
    which expresses its dc floor in this caller's normalisation. The pair
    therefore reproduces the centre strip's dc resistance $1/(\sigma WT)$
    exactly at dc and Cohn's $R_s k_c$ exactly under strong skin effect, with
    a single term across the whole band. An unspecified thickness keeps the
    historical behaviour instead: :class:`HalfSpaceSurfaceImpedance` on a zero
    weight, so conductor loss vanishes entirely. See **The unspecified
    thickness** below.

    **Where the dc resistance comes from, for stripline**

    The dc floor is the *centre strip's* resistance alone, $1/(\sigma WT)$
    over the physical width $W$ and thickness $T$, with no contribution from
    the ground planes. Three stripline-specific assumptions stand behind
    that, and none of them is inherited from the microstrip argument:

    1. **The strip carries its current uniformly at dc.** With no skin effect
       the strip cross-section is an equipotential-driven resistor of area
       $WT$. The fringing-corrected width $W_e$ that
       :class:`CohnStriplineFormulation` uses is an *electromagnetic* width
       fitted to reproduce $Z_c$; it is not a conduction area, and it is
       deliberately not used here. The cross-section record supplies the
       physical $W$.
    2. **The return path is two ground planes of unbounded extent.** Cohn's
       analysis places the strip midway between infinite parallel planes, and
       the closed forms above inherit that geometry. An unbounded sheet has
       no per-unit-length dc resistance -- the return current spreads without
       limit transverse to the line -- so the planes contribute nothing to
       the dc floor. This is a statement about Cohn's geometry, not an
       approximation chosen here: a real stripline's plane resistance depends
       on its finite extent and copper weight, neither of which is a
       cross-section input.
    3. **The split between the two planes is even.** The cross-section is
       symmetric about the strip, so each plane returns half the current by
       construction, and no strip/ground asymmetry of the kind that motivates
       :class:`~pmrf.models.components.lines.microstrip.TraceGroundCurrentDistribution`
       arises. Whether stripline should nevertheless charge plane loss on a
       separate weight at high frequency is a live question and is *not*
       settled here; this strategy stays on Cohn's single pair.

    **The dc-to-skin-effect transition policy**

    There is no separate dc term to add, and so nothing to double-count. The
    dc resistance enters *inside* the surface impedance, as the dc limit of
    the even slab mode divided by this weight, and the same expression tends
    to $\zeta_c$ under strong skin effect. Cohn's weight multiplies the one
    term at both ends of the band, which is what makes the two limits exact
    simultaneously: at dc the $\alpha=1/(2Wk_c)$ factor cancels the weight
    and leaves $1/(\sigma WT)$; at strong skin the factor has gone and Cohn's
    $R_sk_c$ is untouched. The crossover is therefore set by the slab modes'
    own argument $\gamma_c T/2$, at $T\approx\delta$, and is not a fitted or
    switched blend.

    **The unspecified thickness**

    With ``t=None`` the strategy emits a zero weight and
    :class:`HalfSpaceSurfaceImpedance`, so the line has no conductor loss at
    all. This is intentional and unchanged. Cohn's $\alpha_c$ expressions both
    contain $\log(1/T)$ terms and diverge as $T\to0$: a zero-thickness strip
    has no defined conductor loss in this model, at dc or anywhere else, and
    a dc floor $1/(\sigma WT)$ is likewise infinite there. Supplying a
    thickness is what makes conductor loss meaningful, and the zero weight
    says so rather than silently substituting a thickness Cohn's formulas
    never saw.

    **Validity**

    Cohn's $\alpha_c$ is fitted for a thin strip centred between the planes;
    the weight's accuracy degrades as $T/b$ grows, and above roughly
    $T/b=0.4$ the resulting $\alpha=1/(2Wk_c)$ can exceed 1, where
    :class:`~pmrf.materials.surface_impedance.EvenOddSlabSurfaceImpedance`
    itself becomes non-monotone. The switch between Cohn's two $\alpha_c$
    expressions at $\sqrt{\varepsilon_r}\,\Re(Z_c)=120$ is a discontinuity in
    the weight, inherited from the published form.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE
    Transactions on Microwave Theory and Techniques, 3(2), 119-126.

    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.7. Wiley.

    Holloway, C. L., & Kuester, E. F. (1994). Edge shape effects and
    quasi-closed form expressions for the conductor loss of microstrip
    lines. Radio Science, 29(3), 539-559. Eq. (45).
    """

    cross_section_type: ClassVar[type] = StriplineCrossSection

    #: Finite-thickness surface impedance for the centre strip, used whenever
    #: the thickness is known. The default matches both the dc resistance and
    #: Cohn's strong-skin result under Cohn's weight, and puts the conductor's
    #: internal reactance on the slab's $\omega$ law rather than the
    #: semi-infinite $\sqrt{\omega}$ one. See
    #: :class:`~pmrf.materials.surface_impedance.AbstractSurfaceImpedance` for
    #: normalisation details, and pass ``HalfSpaceSurfaceImpedance()`` to
    #: compare against a solver run with a thickness-free skin-effect
    #: approximation.
    slab_impedance: AbstractSurfaceImpedance = eqx.field(
        default_factory=EvenOddSlabSurfaceImpedance
    )

    def _distribute(self, freq, cross_section, quasi_static):
        w, b, t = cross_section.w, cross_section.b, cross_section.t
        if t is None:
            # Cohn's alpha_c diverges as T -> 0, and so does the dc floor;
            # a zero weight says the model has nothing to offer rather than
            # inventing a thickness. See the class docstring.
            return ((HalfSpaceSurfaceImpedance(), jnp.asarray(0.0)),)

        ep_r = jnp.real(cross_section.ep_r)
        zc_real = jnp.real(quasi_static.zc)
        a = 1 + 2 * w / (b - t) + (b + t) / (jnp.pi * (b - t)) * jnp.log((2 * b - t) / t)
        alpha_low = 2.7e-3 * ep_r * zc_real / (30 * jnp.pi * (b - t)) * a
        beta = 1 + b / (0.5 * w + 0.7 * t) * (
            0.5 + 0.7 * t / w + jnp.log(4 * jnp.pi * w / t) / (2 * jnp.pi)
        )
        alpha_high = 0.16 / (zc_real * b) * beta
        alpha_over_rs = jnp.where(jnp.sqrt(ep_r) * zc_real < 120, alpha_low, alpha_high)
        weight = 2 * alpha_over_rs * zc_real
        return ((self.slab_impedance, weight),)


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
    :class:`CohnCurrentDistribution`,
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


class StriplineLine(AbstractImmittanceLine):
    r"""
    Stripline defined by its geometry and material modules.

    The default is :class:`CohnStriplineFormulation`. Homogeneous filling gives
    $\varepsilon_e=\varepsilon_r$ without a separate modal-dispersion model.
    Material dispersion remains available through the dielectric.

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
            conductor=BulkConductor(sigma=5.8e7),
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
        Thickness of the centre strip in meters. A known thickness gives the
        line a dc series resistance of $1/(\sigma WT)$ and a finite-thickness
        conductor loss; ``None`` idealises it as zero-thickness, which has no
        finite conductor loss at any frequency. See
        :class:`CohnCurrentDistribution`.
    dielectric : AbstractDielectric, default=ConstantDielectric(ep_r=4.3)
        The filling between the ground planes. A scalar permittivity or an
        ``(ep_r, tand)`` tuple is coerced into a
        :class:`~pmrf.materials.ConstantDielectric`.
    conductor : AbstractConductor, default=BulkConductor()
        The material of the strip and the ground planes. A scalar conductivity in
        S/m is coerced into a :class:`~pmrf.materials.BulkConductor`.
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

    #: The conductor current-distribution strategy
    current_distribution: AbstractCurrentDistribution = field(
        default_factory=CohnCurrentDistribution
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
        return quasi_static.to_immittance(
            freq, dielectric, conductor,
            current_distribution=self.current_distribution,
            cross_section=StriplineCrossSection(
                w=self.w, b=self.b, t=self.t, ep_r=dielectric.ep_r
            ),
        )
