r"""
Conductor cross-section shapes.

A shape formulation is the layer between a conductor material and a line's
geometry. Given the metal's intrinsic surface impedance
$\zeta_c=\sqrt{j\omega\mu/\sigma}$ and a cross-section shape, it answers one
question: what is the surface impedance, in ohm per square? Every entry
returns $\zeta_c$ times a dimensionless shape factor; the geometry weight
that turns this into a per-unit-length series impedance (e.g. $1/2\pi a$ for
a round conductor) is supplied by the caller, not folded in here. This is
what lets a coaxial shield and a microstrip trace share the same
finite-thickness physics.
"""
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import mu_0

from pmrf.materials.properties import ConductorProperties
from pmrf.math.bessel import i0_over_i1, i1e, k0_over_k1, k1e


class AbstractConductorShape(eqx.Module):
    r"""
    Abstract base class for a conductor cross-section shape formulation.

    A shape formulation is pure numerics, like a line
    :mod:`~pmrf.models.components.lines.formulations` strategy: every
    argument arrives as an already-evaluated array, so it can be checked
    directly against the equations of the paper it comes from with no
    ParamRF objects in sight. Concrete shapes differ in the geometry they
    need -- a rod takes a radius, a tube a radius and a wall thickness, a
    half-space none -- so only the common material argument is fixed here.

    **Normalisation convention**

    Every entry returns an impedance in ohm per square, referred to the
    current its own cross-section carries, and the caller multiplies by an
    inverse-metre geometry weight $k$ to obtain a per-unit-length series
    impedance. The round entries and the caller agree on that normalisation
    exactly: :class:`SchelkunoffRodShape` is referred to the total rod
    current and the caller's weight is $1/2\pi a$, so the product is the
    rod's true impedance at every frequency.

    The planar entries do not have that luxury, because the planar weight in
    use is Wheeler's incremental-inductance factor, which is a different
    decomposition of the problem. :class:`HollowayKuesterSlabShape` is
    referred to the *total strip current*, so reproducing the true dc
    resistance $1/(\sigma W t)$ from its dc value $2/(\sigma t)$ requires a
    weight of $1/(2W)$ -- for a 1.55 mm trace that is $322.6\,\mathrm{m^{-1}}$
    against Wheeler's $963.7\,\mathrm{m^{-1}}$, a factor of **2.99**. The
    discrepancy is real physics, not a scale error: Wheeler's weight is
    larger because it is an incremental-inductance result covering the
    ground plane and the edge crowding as well as the strip, while the slab
    is a one-dimensional strip diffusion result containing neither. Under
    Wheeler's weight the exact slab therefore returns 0.925 ohm/m at dc for
    35 um copper on that trace where the true value is 0.3097 ohm/m, and no
    rescaling repairs it: halving the slab to fix dc breaks the strong-skin
    asymptote by the same factor.

    A frequency-independent weight consequently admits only one entry that
    is right at both ends, :class:`RootSumSquareSlabShape`, which reaches
    into the caller's normalisation through the optional ``weight``
    argument. That is why the blend is the planar default and not merely an
    industry convention; the honest fix is separate strip and ground-plane
    weights (Holloway--Kuester), which ParamRF does not implement.
    """
    @abstractmethod
    def impedance(self, omega, conductor: ConductorProperties, **geometry) -> jnp.ndarray:
        r"""
        Return the surface impedance of this shape, in ohm per square.

        Parameters
        ----------
        omega : ArrayLike
            Angular frequency in rad/s. The frequency argument is *not*
            named ``w``: under the symbol-named geometry convention ``w`` is
            a strip width, which a planar entry takes at the same call.
        conductor : ConductorProperties
            The metal's evaluated properties. ``conductor.zs`` is the
            surface prefactor -- for a smooth bulk metal the intrinsic
            surface impedance $\zeta_c=\sqrt{j\omega\mu/\sigma}$, but a
            surface treatment such as roughness scales it -- and
            ``conductor.gamma(omega)`` is the diffusion constant
            $\gamma=\sqrt{j\omega\mu\sigma}$ inside the bulk, which supplies
            the dimensionless $\gamma a$ and $\gamma t$ arguments. The two
            are independent inputs: never recover one from the other via
            $\gamma=\sigma\zeta_c$, an identity that holds only for a smooth
            bulk conductor. ``zs`` is not yet weighted by this shape's
            factor.
        **geometry
            Shape-specific cross-section dimensions, in meters. A shape
            accepts and ignores dimensions it has none of -- a half-space
            takes a radius and a wall thickness and uses neither -- so a
            caller choosing between shapes never has to know which
            arguments a particular one reads.
        weight : ArrayLike, optional
            The inverse-metre geometry weight the caller is about to
            multiply this entry's answer by, passed as part of the geometry
            so that an entry whose dc floor is fixed in per-unit-length
            terms can express that floor in the caller's normalisation. Only
            :class:`RootSumSquareSlabShape` reads it; see the normalisation
            note in :class:`AbstractConductorShape`.

        Returns
        -------
        jnp.ndarray
            Surface impedance in ohm per square.
        """
        raise NotImplementedError


class HalfSpaceShape(AbstractConductorShape):
    r"""
    Leontovich half-space boundary: the trivial shape factor.

    **Mathematical Formulation**

    $$Z_s = \zeta_c$$

    A locally flat conductor carries no cross-section of its own, so the
    surface impedance is the metal's intrinsic impedance, unweighted. Every
    other shape in this layer approaches this one in the strong-skin limit,
    where the skin depth is small compared to the radius of curvature.

    **Validity**

    Exact for a genuine half-space, and a good approximation for any surface
    whose radius of curvature is large compared to the skin depth -- which is
    also the regime every curved shape below converges to.

    References
    ----------
    Leontovich, M. A. (1948). Approximate boundary conditions for the
    electromagnetic field on the surface of a well-conducting medium, in
    Investigations of Radiowave Propagation, Part II, 5-12. Academy of
    Sciences, USSR.
    """
    def impedance(self, omega, conductor: ConductorProperties, **geometry) -> jnp.ndarray:
        # Cross-section dimensions are accepted and ignored: a half-space has
        # none, and a caller choosing between shapes should not have to know
        # which of them take a radius or a wall thickness.
        return conductor.zs


class HollowayKuesterSlabShape(AbstractConductorShape):
    r"""
    Exact finite-thickness impedance for the total current in a planar strip.

    **Mathematical Formulation**

    Holloway and Kuester's eq. (45) gives the coupled impedances of the two
    strip faces, $Z_s=\zeta_c\coth(\gamma_c t)$ and
    $Z_m=\zeta_c\operatorname{csch}(\gamma_c t)$.  Their eq. (100) shows
    that the total current therefore sees
    $$Z_s+Z_m=\zeta_c\coth(\gamma_c t/2),\qquad
    \gamma_c=\sqrt{j\omega\mu\sigma}.$$

    The half-thickness argument is essential: the scalar represents total
    current, rather than the self impedance of either face.  Its limits are
    $2/(\sigma t)$ at dc and $\zeta_c$ in strong skin effect.

    **Validity**

    Neglecting the difference-current impedance
    $\zeta_c\tanh(\gamma_c t/2)$ is valid for quasi-TEM coplanar waveguide
    and for microstrip whose characteristic impedance exceeds 40 ohm.

    **Normalisation**

    The scalar is referred to the *total* strip current, so the geometry
    weight that turns it into a per-unit-length resistance is $1/(2W)$: at
    dc the entry returns $2/(\sigma t)$ and $2/(\sigma t)\cdot 1/(2W)$ is
    the true $1/(\sigma W t)$. That is not the weight
    :class:`~pmrf.models.components.lines.current_distribution.WheelerCurrentDistribution`
    supplies -- for a 1.55 mm trace Wheeler's weight is
    $963.7\,\mathrm{m^{-1}}$ against $1/(2W)=322.6\,\mathrm{m^{-1}}$, a
    factor of 2.99 -- because Wheeler's weight also covers the ground plane
    and the edge crowding that this one-dimensional strip result does not
    contain. Charged with Wheeler's weight the entry therefore gives
    0.925 ohm/m at dc for 35 um copper on that trace where the true value is
    0.3097 ohm/m. Rescaling cannot fix both ends at once, which is why
    :class:`RootSumSquareSlabShape` and not this entry is the planar
    default; see the normalisation note on
    :class:`AbstractConductorShape`. Use this entry with a weight it is
    normalised against.

    References
    ----------
    Holloway, C. L., & Kuester, E. F. (1994). Edge shape effects and
    quasi-closed form expressions for the conductor loss of microstrip
    lines. Radio Science, 29(3), 539-559. Eq. (45), (100).

    Rautio, J. C., & Demir, V. (2003). Microstrip Conductor Loss Models for
    Electromagnetic Analysis. IEEE Transactions on Microwave Theory and
    Techniques, 51(3), 915-921. Eq. (4).
    """
    def impedance(self, omega, conductor: ConductorProperties, *, t, **geometry) -> jnp.ndarray:
        gamma_t_over_two = conductor.gamma(omega) * t / 2
        evaluable = (omega > 0) & jnp.isfinite(conductor.sigma)
        safe_argument = jnp.where(evaluable, gamma_t_over_two, 1.0)
        zs = conductor.zs / jnp.tanh(safe_argument)
        dc = 2 / (conductor.sigma * t)
        return jnp.where(evaluable, zs, dc)


class RootSumSquareSlabShape(AbstractConductorShape):
    r"""
    ParamRF's smooth resistance-only blend for a finite planar conductor.

    **Mathematical Formulation**

    $$Z_s=\sqrt{R_{dc,sq}^2+\Re(\zeta_c)^2}+j\Im(\zeta_c),\qquad
    R_{dc,sq}=\frac{1}{\sigma W t\,k},$$

    where $W$ is the strip width, $t$ its thickness and $k$ the caller's
    inverse-metre geometry weight. The entry computes its own dc floor from
    those dimensions: charged with $k$ the first term becomes exactly the
    strip's true dc resistance $1/(\sigma W t)$, and the second becomes the
    semi-infinite result $k\,\Re(\zeta_c)$. This is a ParamRF convention and
    has no source paper.

    **Why it is the planar default**

    Under a frequency-independent geometry weight this is the *only* entry
    that is right at both ends. :class:`HollowayKuesterSlabShape` is exact
    but is normalised to the total strip current, which fixes its weight at
    $1/(2W)$ -- a factor of 2.99 away from Wheeler's for a 1.55 mm trace --
    so it cannot satisfy the dc asymptote under the weight actually in use;
    see the normalisation note on :class:`AbstractConductorShape`. Reaching
    into the caller's normalisation through ``weight`` is the price of
    satisfying both asymptotes at once, and is why the industry tools
    converged on the same blend.

    **Validity**

    Against the exact slab over 10--500 MHz its transition-shape residual
    after the best global scale is 11% for a 1.55 mm by 35 um trace, 14% for
    0.45 mm by 35 um, and 15% for a 0.35 mm by 35 um, 96-ohm trace.

    The blend is resistance-only: it keeps the half-space reactance
    $\Im(\zeta_c)$, which grows as $\sqrt{\omega}$, where the true finite
    slab's internal inductance saturates at $\omega\mu t/6$. Below
    $t\approx\delta$ it is therefore not merely approximate in the
    transition but qualitatively wrong on internal inductance. Measured
    against :class:`HollowayKuesterSlabShape` for 35 um copper:

    ==================  ======  ======  ======  ======  ======
    quantity            100kHz    1MHz    5MHz   10MHz   50MHz
    ==================  ======  ======  ======  ======  ======
    $t/\delta$            0.17    0.54    1.20    1.70    3.79
    blend $X$/exact $X$  17.7x    5.6x    2.5x    1.8x   1.01x
    ==================  ======  ======  ======  ======  ======

    The honest fix is separate strip and ground-plane weights, which
    Holloway--Kuester supply and ParamRF does not implement.

    References
    ----------
    ParamRF convention; no source paper.

    Holloway, C. L., & Kuester, E. F. (1994). Edge shape effects and
    quasi-closed form expressions for the conductor loss of microstrip
    lines. Radio Science, 29(3), 539-559.
    """
    def impedance(self, omega, conductor: ConductorProperties, *, w, t, weight, **geometry) -> jnp.ndarray:
        # The dc floor is a per-unit-length quantity, 1/(sigma*W*t), while
        # this layer returns ohm per square: dividing by the caller's weight
        # places that floor in the caller's normalisation, so that
        # multiplying by the weight restores it exactly.
        r_dc_sq = 1 / (conductor.sigma * w * t * weight)
        resistance = jnp.sqrt(r_dc_sq**2 + jnp.real(conductor.zs) ** 2)
        return resistance + 1j * jnp.imag(conductor.zs)


def _tesche_circuit_impedance(zeta_c, r_dc_sq, inverse_l_int_sq, omega):
    """Blend Tesche's dc and high-frequency limits through his equivalent circuit.

    The internal inductance arrives inverted so that an infinite one -- an
    infinitely thick tube wall -- enters as a plain zero. Complex arithmetic
    on an infinity does not survive: ``1j*omega*inf`` is ``nan + infj``, and
    dividing a complex number by ``inf + 0j`` is nan under XLA's division.
    """
    safe_omega = jnp.where(omega > 0, omega, 1.0)
    z = r_dc_sq + zeta_c / (1 + zeta_c * inverse_l_int_sq / (1j * safe_omega))
    return jnp.where(omega > 0, z, r_dc_sq)


class TescheRodShape(AbstractConductorShape):
    r"""
    Solid round conductor, via Tesche's equivalent circuit.

    **Mathematical Formulation**

    Tesche's circuit blends the exact per-unit-length dc resistance and
    internal inductance of a round conductor,
    $$R_{dc} = \frac{1}{\pi a^2\sigma},\qquad L_{int} = \frac{\mu}{8\pi},$$
    into $Z = R_{dc} + \zeta_c/2\pi a\big/[1 + (\zeta_c/2\pi a)/(j\omega
    L_{int})]$. This layer returns the equivalent surface impedance
    $2\pi a Z$, so that the caller's own $1/2\pi a$ geometry weight
    reproduces $Z$ exactly:
    $$Z_s = R_{dc,sq} + \frac{\zeta_c}{1+\zeta_c/(j\omega L_{int,sq})},
    \qquad R_{dc,sq}=\frac{2}{a\sigma},\quad L_{int,sq}=\frac{\mu a}{4}.$$

    **Validity**

    Interpolates between the exact dc resistance and the bare half-space
    impedance $\zeta_c$, so it carries no geometric fit range. It is not
    exact at finite frequency, and it does not reach the exact
    high-frequency limit of a round conductor either: the true strong-skin
    expansion of :class:`SchelkunoffRodShape` is
    $\zeta_c(1 + 1/2\gamma a + \ldots)$, and the circuit never produces
    that $1/2\gamma a$ curvature term. Its own strong-skin limit is instead
    $\zeta_c + R_{dc,sq}$ -- a known, persistent defect of the circuit
    approximation, not a porting error (see ``tests/test_materials/
    test_conductor_shape.py`` for the size of the departure). The error is
    a *shape* error, so refitting $\sigma$ does not absorb it; prefer
    :class:`SchelkunoffRodShape` unless the Bessel evaluation is too
    expensive.

    References
    ----------
    Tesche, F. M. (2007). A Simple Model for the Line Parameters of a Lossy
    Coaxial Cable Filled With a Nondispersive Dielectric. IEEE Transactions
    on Electromagnetic Compatibility, 49(1), 12-17.
    """
    def impedance(self, omega, conductor: ConductorProperties, *, a, **geometry) -> jnp.ndarray:
        r_dc_sq = 2 / (a * conductor.sigma)
        inverse_l_int_sq = 4 / (mu_0 * conductor.mu_r * a)
        return _tesche_circuit_impedance(conductor.zs, r_dc_sq, inverse_l_int_sq, omega)


class TescheTubeShape(AbstractConductorShape):
    r"""
    Tubular conductor of finite wall thickness, via Tesche's equivalent circuit.

    **Mathematical Formulation**

    With $q=(a/(a+t))^2$, the tube's per-unit-length dc resistance and
    internal inductance are
    $$R_{dc} = \frac{1}{2\pi a t\sigma},\qquad
    L_{int} = \frac{\mu}{2\pi}\left[\frac{\ln(1+t/a)}{(1-q)^2}
    + \frac{q-3}{4(1-q)}\right],$$
    blended into $Z$ the same way as the solid rod. This layer returns
    $2\pi a Z$:
    $$Z_s = R_{dc,sq} + \frac{\zeta_c}{1+\zeta_c/(j\omega L_{int,sq})},
    \qquad R_{dc,sq}=\frac{1}{t\sigma},\quad
    L_{int,sq}=\mu a\left[\frac{\ln(1+t/a)}{(1-q)^2}+\frac{q-3}{4(1-q)}\right].$$

    **Validity**

    Same circuit approximation as the solid rod, so it shares the same
    defect: its strong-skin limit is $\zeta_c + R_{dc,sq}$, not $\zeta_c$
    exactly. An infinite $t$ is the default, and is taken analytically
    rather than arithmetically: $R_{dc,sq}\to0$ and $L_{int,sq}\to\infty$,
    so the shape reduces exactly to :class:`HalfSpaceShape`. That is how
    :class:`~pmrf.models.components.lines.formulations.TescheCoaxialFormulation`
    describes an outer shield whose wall thickness is not modelled -- it
    passes an infinite $t$ rather than swapping the shape.

    References
    ----------
    Tesche, F. M. (2007). A Simple Model for the Line Parameters of a Lossy
    Coaxial Cable Filled With a Nondispersive Dielectric. IEEE Transactions
    on Electromagnetic Compatibility, 49(1), 12-17.
    """
    def impedance(self, omega, conductor: ConductorProperties, *, a, t=jnp.inf, **geometry) -> jnp.ndarray:
        # An infinite wall is taken analytically rather than arithmetically:
        # both the dc resistance and the inverse internal inductance are zero
        # there, and evaluating the expressions at t = inf would leave an
        # inf/inf behind instead.
        finite_wall = jnp.isfinite(t)
        safe_t = jnp.where(finite_wall, t, a)
        q = (a / (a + safe_t)) ** 2
        r_dc_sq = jnp.where(finite_wall, 1 / (safe_t * conductor.sigma), 0.0)
        l_int_sq = mu_0 * conductor.mu_r * a * (
            jnp.log1p(safe_t / a) / (1 - q) ** 2 + (q - 3) / (4 * (1 - q))
        )
        inverse_l_int_sq = jnp.where(finite_wall, 1 / l_int_sq, 0.0)
        return _tesche_circuit_impedance(conductor.zs, r_dc_sq, inverse_l_int_sq, omega)


class SchelkunoffRodShape(AbstractConductorShape):
    r"""
    Solid round conductor, exact.

    **Mathematical Formulation**

    Schelkunoff solves Maxwell's equations inside the rod rather than
    interpolating between its limits. His eq. (65), the internal impedance
    per unit length of a solid cylinder of radius $a$, is
    $$Z = \frac{\gamma}{2\pi a\sigma}\,\frac{I_0(\gamma a)}{I_1(\gamma a)},
    \qquad \gamma=\sqrt{j\omega\mu\sigma}.$$
    For a smooth bulk metal $\zeta_c=\sqrt{j\omega\mu/\sigma}=\gamma/\sigma$,
    so the shape factor this layer returns -- with the caller supplying its
    own $1/2\pi a$ geometry weight -- is simply the Bessel ratio:
    $$Z_s = \zeta_c\,\frac{I_0(\gamma a)}{I_1(\gamma a)}.$$

    Both limits follow with no special case. As $\gamma a\to0$,
    $I_0/I_1\to2/\gamma a$ and $Z_s\to2/a\sigma$, which is the exact dc
    sheet resistance $\pi a^2$ times $1/\pi a^2\sigma$; as
    $\gamma a\to\infty$, $I_0/I_1\to1+1/2\gamma a$, so the shape
    approaches :class:`HalfSpaceShape` with the leading curvature
    correction that :class:`TescheRodShape` cannot reproduce.

    **Validity**

    Exact for a homogeneous, isotropic, non-magnetic-hysteresis solid rod
    carrying axially symmetric current, at every frequency. The remaining
    error is the numerics of :func:`~pmrf.math.bessel.i0_over_i1`, below
    3.1e-8 relative on the $\gamma$ ray.

    References
    ----------
    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial
    Transmission Lines and Cylindrical Shields. Bell System Technical
    Journal, 13(4), 532-579. Eq. (65).
    """
    def impedance(self, omega, conductor: ConductorProperties, *, a, **geometry) -> jnp.ndarray:
        # gamma vanishes at dc, where the ratio has a pole that exactly
        # cancels zeta_c's zero. Take that limit analytically and keep the
        # argument away from the pole so the gradient stays finite. A
        # perfect conductor is the other unevaluable argument: gamma is
        # infinite there, but zeta_c is zero, so the shape factor never has
        # to be evaluated at all -- which is why the final guard below is
        # the narrower `w > 0`: an infinite sigma needs a safe Bessel
        # argument, but its zero zeta_c already gives the right answer, and
        # 2/(a*sigma) is likewise zero there.
        gamma_a = conductor.gamma(omega) * a
        evaluable = (omega > 0) & jnp.isfinite(conductor.sigma)
        safe_gamma_a = jnp.where(evaluable, gamma_a, 1.0)
        zs = conductor.zs * i0_over_i1(safe_gamma_a)
        r_dc_sq = 2 / (a * conductor.sigma)
        return jnp.where(omega > 0, zs, r_dc_sq)


class SchelkunoffTubeShape(AbstractConductorShape):
    r"""
    Cylindrical tube of finite wall thickness, exact.

    **Mathematical Formulation**

    For inner radius $a$, outer radius $b=a+t$ and current returning on the
    **inner** surface, Schelkunoff's eq. (74) gives the internal impedance
    referred to that surface. Written as a shape factor -- the caller
    supplies the $1/2\pi a$ geometry weight -- it is
    $$Z_s = \zeta_c\,
    \frac{I_0(\gamma a)K_1(\gamma b)+K_0(\gamma a)I_1(\gamma b)}
    {I_1(\gamma b)K_1(\gamma a)-I_1(\gamma a)K_1(\gamma b)},
    \qquad\gamma=\sqrt{j\omega\mu\sigma}.$$

    Dividing through by $I_1(\gamma b)K_1(\gamma a)$ turns this into three
    bounded pieces,
    $$Z_s = \zeta_c\,\frac{K_0(\gamma a)/K_1(\gamma a)
    + \big[I_0(\gamma a)/I_1(\gamma a)\big]\,\rho}{1-\rho},
    \qquad
    \rho = \frac{I_1(\gamma a)K_1(\gamma b)}{I_1(\gamma b)K_1(\gamma a)},$$
    which is what is evaluated. The two ratios are
    :func:`~pmrf.math.bessel.i0_over_i1` and
    :func:`~pmrf.math.bessel.k0_over_k1`; the surface-coupling term $\rho$ is
    assembled from the exponentially scaled
    :func:`~pmrf.math.bessel.i1e` and :func:`~pmrf.math.bessel.k1e`, so that
    the only exponential left is $e^{-2\gamma t}$, which never exceeds unity.
    No Bessel function is ever evaluated unscaled, and nothing overflows for
    a thick wall.

    Both limits fall out with no special case. In strong skin effect
    $\rho\to e^{-2\gamma t}$ and the shape becomes
    $\zeta_c\coth(\gamma t)$ with the cylindrical curvature corrections
    carried by the two ratios. As $\gamma\to0$, $\rho\to a^2/b^2$ and
    $I_0/I_1\to2/\gamma a$, leaving the exact dc sheet resistance
    $$Z_s \to \frac{2a}{\sigma(b^2-a^2)} = \frac{1}{\sigma t}
    \cdot\frac{2a}{a+b},$$
    which is $2\pi a$ times $1/\sigma\pi(b^2-a^2)$, the tube's dc resistance
    per unit length, and reduces to $1/\sigma t$ for a thin wall. That is the
    value the $\omega=0$ branch below returns, so the dc value and the
    low-frequency limit agree exactly rather than differing by a factor of
    two.

    An infinite ``t`` is accepted and gives the infinite-wall limit
    $\zeta_c K_0(\gamma a)/K_1(\gamma a)$, which is how a coaxial shield
    whose wall is not modelled is described.

    **Validity**

    Exact for a homogeneous, isotropic tube carrying axially symmetric
    current returning on its inner surface, at every frequency. The
    remaining error is the numerics of the Bessel evaluations, below 1e-7
    relative against :func:`scipy.special` over the weak-skin, transition
    and strong-skin regimes; see ``tests/test_materials/
    test_conductor_shape.py``.

    References
    ----------
    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial
    Transmission Lines and Cylindrical Shields. Bell System Technical
    Journal, 13(4), 532-579. Eq. (74).
    """
    def impedance(self, omega, conductor: ConductorProperties, *, a, t=jnp.inf, **geometry) -> jnp.ndarray:
        # Same guards as the solid rod: gamma vanishes at dc and diverges for
        # a perfect conductor, so both branches are evaluated on arguments
        # the other regime cannot poison. An infinite wall is the third
        # unevaluable argument -- the Bessel functions at gamma*b are then
        # 0 and inf, whose product is NaN -- so it too gets a safe argument
        # and an analytic value, here rho = 0.
        gamma = conductor.gamma(omega)
        evaluable = (omega > 0) & jnp.isfinite(conductor.sigma)
        safe_gamma = jnp.where(evaluable, gamma, 1.0)
        finite_wall = jnp.isfinite(t)
        safe_t = jnp.where(finite_wall, t, a)

        xa = safe_gamma * a
        xb = safe_gamma * (a + safe_t)
        rho = (i1e(xa) * k1e(xb)) / (i1e(xb) * k1e(xa)) * jnp.exp(-2 * safe_gamma * safe_t)
        rho = jnp.where(finite_wall, rho, 0.0)

        z = conductor.zs * (k0_over_k1(xa) + i0_over_i1(xa) * rho) / (1 - rho)
        # sigma divides last so that a perfect conductor sends the whole
        # quotient to zero rather than leaving inf/inf in the wall gradient.
        r_dc_sq = jnp.where(
            finite_wall, (2 * a / (safe_t * (safe_t + 2 * a))) / conductor.sigma, 0.0
        )
        return jnp.where(evaluable, z, r_dc_sq)
