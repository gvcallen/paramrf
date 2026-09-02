r"""
Surface-impedance formulations for conductor cross-sections.

Shape formulations convert material properties and cross-section dimensions
into surface impedance in ohm per square. The caller supplies the geometry
weight that converts this value to per-unit-length series impedance.
"""
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import mu_0

from pmrf.materials.properties import ConductorProperties
from pmrf.math.bessel import i0_over_i1, i1e, k0_over_k1, k1e


class AbstractSurfaceImpedance(eqx.Module):
    r"""
    Abstract base class for a conductor surface-impedance formulation.

    Shape formulations operate on evaluated arrays. Required geometry varies
    by shape, so only the material argument is common to the interface.

    **Normalisation convention**

    Each formulation returns an impedance in ohm per square. The caller
    multiplies it by a geometry weight $k$ in inverse metres to obtain the
    per-unit-length series impedance. For example,
    :class:`SchelkunoffRodShape` is referred to the total rod current and is
    paired with $k=1/(2\pi a)$.

    Planar formulations require care because Wheeler's
    incremental-inductance weight includes the ground plane and edge-current
    crowding. By contrast, :class:`HollowayKuesterSlabShape` describes
    one-dimensional diffusion in the strip and is normalised to the total
    strip current. Its dc limit requires $k=1/(2W)$; using Wheeler's weight
    cannot reproduce both the dc resistance and strong-skin asymptote.
    :class:`RootSumSquareSlabShape` uses the optional ``weight`` argument to
    match both limits and is therefore the planar default. A full
    Holloway--Kuester treatment would require separate strip and ground-plane
    weights, which ParamRF does not implement.
    """
    @abstractmethod
    def impedance(self, omega, conductor: ConductorProperties, **geometry) -> jnp.ndarray:
        r"""
        Return the surface impedance of this shape, in ohm per square.

        Parameters
        ----------
        omega : ArrayLike
            Angular frequency in rad/s. ``w`` is reserved for strip width.
        conductor : ConductorProperties
            Evaluated metal properties. ``conductor.zs`` is the surface
            prefactor, including any surface treatment, and
            ``conductor.gamma(omega)`` is the bulk diffusion constant. Treat
            them as independent: $\gamma=\sigma\zeta_c$ holds only for a
            smooth bulk conductor.
        **geometry
            Shape-specific dimensions in metres. Implementations ignore
            dimensions they do not use.
        weight : ArrayLike, optional
            Caller's geometry weight in inverse metres. Only
            :class:`RootSumSquareSlabShape` uses it to express its dc limit
            in the caller's normalisation.

        Returns
        -------
        jnp.ndarray
            Surface impedance in ohm per square.
        """
        raise NotImplementedError


class HalfSpaceShape(AbstractSurfaceImpedance):
    r"""
    Leontovich half-space boundary.

    **Mathematical Formulation**

    $$Z_s = \zeta_c$$

    **Validity**

    Exact for a half-space. Approximates a curved surface when its radius of
    curvature is large relative to the skin depth.

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


class HollowayKuesterSlabShape(AbstractSurfaceImpedance):
    r"""
    Exact finite-thickness impedance for the total current in a planar strip.

    **Mathematical Formulation**

    Holloway and Kuester's eq. (45) gives the coupled impedances of the two
    strip faces, $Z_s=\zeta_c\coth(\gamma_c t)$ and
    $Z_m=\zeta_c\operatorname{csch}(\gamma_c t)$.  Their eq. (100) shows
    that the total current therefore sees
    $$Z_s+Z_m=\zeta_c\coth(\gamma_c t/2),\qquad
    \gamma_c=\sqrt{j\omega\mu\sigma}.$$

    The half-thickness argument follows from referring the impedance to total
    strip current. The limits are $2/(\sigma t)$ at dc and $\zeta_c$ under
    strong skin effect.

    **Validity**

    Neglecting the difference-current impedance
    $\zeta_c\tanh(\gamma_c t/2)$ is valid for quasi-TEM coplanar waveguide
    and for microstrip whose characteristic impedance exceeds 40 ohm.

    **Normalisation**

    The impedance is referred to total strip current and therefore requires
    a geometry weight of $1/(2W)$. This differs from the weight supplied by
    :class:`~pmrf.models.components.lines.microstrip.WheelerCurrentDistribution`
    because Wheeler's model also includes the ground plane and edge-current
    crowding. Use this formulation only with a compatible weight; see
    :class:`AbstractSurfaceImpedance`.

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


class RootSumSquareSlabShape(AbstractSurfaceImpedance):
    r"""
    Resistance-only blend for a finite planar conductor.

    **Mathematical Formulation**

    $$Z_s=\sqrt{R_{dc,sq}^2+\Re(\zeta_c)^2}+j\Im(\zeta_c),\qquad
    R_{dc,sq}=\frac{1}{\sigma W t\,k},$$

    where $W$ is the strip width, $t$ its thickness, and $k$ the caller's
    geometry weight in inverse metres. Multiplication by $k$ recovers the
    exact dc resistance $1/(\sigma Wt)$ and the half-space high-frequency
    limit. This is a ParamRF convention.

    **Why it is the planar default**

    Unlike :class:`HollowayKuesterSlabShape`, this formulation uses the
    caller's ``weight`` and therefore matches both asymptotes under a
    frequency-independent planar weight. See :class:`AbstractSurfaceImpedance`.

    **Validity**

    This is not an exact transition model. It retains the half-space
    reactance $\Im(\zeta_c)\propto\sqrt{\omega}$, whereas a finite slab's
    low-frequency reactance approaches $\omega\mu t/6$. Internal inductance
    is therefore inaccurate when the thickness is comparable to or less
    than the skin depth.

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
    """Evaluate Tesche's equivalent-circuit blend.

    Inverse internal inductance represents an infinitely thick wall as zero
    and avoids complex arithmetic with infinity under XLA.
    """
    safe_omega = jnp.where(omega > 0, omega, 1.0)
    z = r_dc_sq + zeta_c / (1 + zeta_c * inverse_l_int_sq / (1j * safe_omega))
    return jnp.where(omega > 0, z, r_dc_sq)


class TescheRodShape(AbstractSurfaceImpedance):
    r"""
    Solid round conductor, via Tesche's equivalent circuit.

    **Mathematical Formulation**

    Tesche's circuit combines the dc resistance and internal inductance,
    $$R_{dc} = \frac{1}{\pi a^2\sigma},\qquad L_{int} = \frac{\mu}{8\pi},$$
    as $Z = R_{dc} + \zeta_c/2\pi a\big/[1 + (\zeta_c/2\pi a)/(j\omega
    L_{int})]$. The equivalent surface impedance $2\pi aZ$ is
    $$Z_s = R_{dc,sq} + \frac{\zeta_c}{1+\zeta_c/(j\omega L_{int,sq})},
    \qquad R_{dc,sq}=\frac{2}{a\sigma},\quad L_{int,sq}=\frac{\mu a}{4}.$$

    **Validity**

    This is an interpolation, not an exact finite-frequency solution. Its
    strong-skin limit is $\zeta_c+R_{dc,sq}$ and omits the
    $1/(2\gamma a)$ curvature term in the exact
    :class:`SchelkunoffRodShape` expansion. Prefer the exact formulation
    unless Bessel evaluation cost is prohibitive.

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


class TescheTubeShape(AbstractSurfaceImpedance):
    r"""
    Tubular conductor of finite wall thickness, via Tesche's equivalent circuit.

    **Mathematical Formulation**

    With $q=(a/(a+t))^2$, the tube's per-unit-length dc resistance and
    internal inductance are
    $$R_{dc} = \frac{1}{2\pi a t\sigma},\qquad
    L_{int} = \frac{\mu}{2\pi}\left[\frac{\ln(1+t/a)}{(1-q)^2}
    + \frac{q-3}{4(1-q)}\right],$$
    combined as for the solid rod. The equivalent surface impedance is
    $$Z_s = R_{dc,sq} + \frac{\zeta_c}{1+\zeta_c/(j\omega L_{int,sq})},
    \qquad R_{dc,sq}=\frac{1}{t\sigma},\quad
    L_{int,sq}=\mu a\left[\frac{\ln(1+t/a)}{(1-q)^2}+\frac{q-3}{4(1-q)}\right].$$

    **Validity**

    This formulation has the same approximation as :class:`TescheRodShape`:
    its strong-skin limit is $\zeta_c+R_{dc,sq}$ rather than $\zeta_c$.
    For $t\to\infty$, $R_{dc,sq}\to0$ and $L_{int,sq}\to\infty$, reducing
    the result to :class:`HalfSpaceShape`. This limit is used by
    :class:`~pmrf.models.components.lines.coaxial.TescheCoaxialFormulation`
    when the outer-shield thickness is unspecified.

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


class SchelkunoffRodShape(AbstractSurfaceImpedance):
    r"""
    Solid round conductor, exact.

    **Mathematical Formulation**

    Schelkunoff's internal impedance per unit length for a cylinder of radius
    $a$ is
    $$Z = \frac{\gamma}{2\pi a\sigma}\,\frac{I_0(\gamma a)}{I_1(\gamma a)},
    \qquad \gamma=\sqrt{j\omega\mu\sigma}.$$
    Since $\zeta_c=\gamma/\sigma$ for a smooth bulk metal, the corresponding
    surface impedance is
    $$Z_s = \zeta_c\,\frac{I_0(\gamma a)}{I_1(\gamma a)}.$$

    As $\gamma a\to0$,
    $I_0/I_1\to2/\gamma a$ and $Z_s\to2/a\sigma$, which is the exact dc
    limit. As
    $\gamma a\to\infty$, $I_0/I_1\to1+1/2\gamma a$, so the shape
    approaches :class:`HalfSpaceShape` with the leading curvature correction.

    **Validity**

    Exact for a homogeneous, isotropic solid rod carrying axially symmetric
    current. Numerical accuracy is limited by
    :func:`~pmrf.math.bessel.i0_over_i1`.

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


class SchelkunoffTubeShape(AbstractSurfaceImpedance):
    r"""
    Cylindrical tube of finite wall thickness, exact.

    **Mathematical Formulation**

    For inner radius $a$, outer radius $b=a+t$, and current returning on the
    inner surface, Schelkunoff's internal impedance is
    $$Z_s = \zeta_c\,
    \frac{I_0(\gamma a)K_1(\gamma b)+K_0(\gamma a)I_1(\gamma b)}
    {I_1(\gamma b)K_1(\gamma a)-I_1(\gamma a)K_1(\gamma b)},
    \qquad\gamma=\sqrt{j\omega\mu\sigma}.$$

    Dividing by $I_1(\gamma b)K_1(\gamma a)$ gives the bounded form used for
    evaluation:
    $$Z_s = \zeta_c\,\frac{K_0(\gamma a)/K_1(\gamma a)
    + \big[I_0(\gamma a)/I_1(\gamma a)\big]\,\rho}{1-\rho},
    \qquad
    \rho = \frac{I_1(\gamma a)K_1(\gamma b)}{I_1(\gamma b)K_1(\gamma a)},$$
    The implementation uses scaled Bessel functions for $\rho$, leaving only
    the decaying exponential $e^{-2\gamma t}$ and avoiding overflow for thick
    walls.

    Under strong skin effect,
    $\rho\to e^{-2\gamma t}$ and the shape becomes
    $\zeta_c\coth(\gamma t)$ with the cylindrical curvature corrections
    retained. As $\gamma\to0$, the exact dc limit is
    $$Z_s \to \frac{2a}{\sigma(b^2-a^2)} = \frac{1}{\sigma t}
    \cdot\frac{2a}{a+b},$$
    which reduces to $1/(\sigma t)$ for a thin wall.

    For ``t=inf``, the result is the infinite-wall limit
    $\zeta_c K_0(\gamma a)/K_1(\gamma a)$.

    **Validity**

    Exact for a homogeneous, isotropic tube carrying axially symmetric
    current that returns on its inner surface. Accuracy is limited by the
    numerical Bessel evaluations.

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
