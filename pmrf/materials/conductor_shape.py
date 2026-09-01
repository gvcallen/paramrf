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
from pmrf.math.bessel import i0_over_i1


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
    """
    @abstractmethod
    def impedance(self, w, conductor: ConductorProperties, **geometry) -> jnp.ndarray:
        r"""
        Return the surface impedance of this shape, in ohm per square.

        Parameters
        ----------
        w : ArrayLike
            Angular frequency in rad/s.
        conductor : ConductorProperties
            The metal's evaluated properties. ``conductor.zs`` is its
            intrinsic surface impedance $\zeta_c=\sqrt{j\omega\mu/\sigma}$,
            not yet weighted by this shape's factor.
        **geometry
            Shape-specific cross-section dimensions, in meters.

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
    def impedance(self, w, conductor: ConductorProperties) -> jnp.ndarray:
        return conductor.zs


def _tesche_circuit_impedance(zeta_c, r_dc_sq, l_int_sq, w):
    """Blend Tesche's dc and high-frequency limits through his equivalent circuit."""
    safe_w = jnp.where(w > 0, w, 1.0)
    z = r_dc_sq + zeta_c / (1 + zeta_c / (1j * safe_w * l_int_sq))
    return jnp.where(w > 0, z, r_dc_sq)


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
    def impedance(self, w, conductor: ConductorProperties, *, a) -> jnp.ndarray:
        r_dc_sq = 2 / (a * conductor.sigma)
        l_int_sq = mu_0 * conductor.mu_r * a / 4
        return _tesche_circuit_impedance(conductor.zs, r_dc_sq, l_int_sq, w)


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
    exactly. In the infinite-wall limit ($t\to\infty$), $R_{dc,sq}\to0$ and
    $L_{int,sq}\to\infty$, and this reduces to :class:`HalfSpaceShape` --
    how :class:`~pmrf.models.components.lines.formulations.TescheCoaxialFormulation`
    treats an outer shield whose wall thickness is not modelled.

    References
    ----------
    Tesche, F. M. (2007). A Simple Model for the Line Parameters of a Lossy
    Coaxial Cable Filled With a Nondispersive Dielectric. IEEE Transactions
    on Electromagnetic Compatibility, 49(1), 12-17.
    """
    def impedance(self, w, conductor: ConductorProperties, *, a, t) -> jnp.ndarray:
        q = (a / (a + t)) ** 2
        r_dc_sq = 1 / (t * conductor.sigma)
        l_int_sq = mu_0 * conductor.mu_r * a * (
            jnp.log1p(t / a) / (1 - q) ** 2 + (q - 3) / (4 * (1 - q))
        )
        return _tesche_circuit_impedance(conductor.zs, r_dc_sq, l_int_sq, w)


class SchelkunoffRodShape(AbstractConductorShape):
    r"""
    Solid round conductor, exact.

    **Mathematical Formulation**

    Schelkunoff solves Maxwell's equations inside the rod rather than
    interpolating between its limits. His eq. (65), the internal impedance
    per unit length of a solid cylinder of radius $a$, is
    $$Z = \frac{\gamma}{2\pi a\sigma}\,\frac{I_0(\gamma a)}{I_1(\gamma a)},
    \qquad \gamma=\sqrt{j\omega\mu\sigma}.$$
    Since $\zeta_c=\sqrt{j\omega\mu/\sigma}=\gamma/\sigma$, the shape
    factor this layer returns -- with the caller supplying its own
    $1/2\pi a$ geometry weight -- is simply the Bessel ratio:
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
    def impedance(self, w, conductor: ConductorProperties, *, a) -> jnp.ndarray:
        # gamma = sigma * zeta_c vanishes at dc, where the ratio has a pole
        # that exactly cancels zeta_c's zero. Take that limit analytically
        # and keep the argument away from the pole so the gradient stays
        # finite. A perfect conductor is the other unevaluable argument:
        # gamma is infinite there, but zeta_c is zero, so the shape factor
        # never has to be evaluated at all -- which is why the final guard
        # below is the narrower `w > 0`: an infinite sigma needs a safe
        # Bessel argument, but its zero zeta_c already gives the right
        # answer, and 2/(a*sigma) is likewise zero there.
        gamma_a = conductor.sigma * conductor.zs * a
        evaluable = (w > 0) & jnp.isfinite(conductor.sigma)
        safe_gamma_a = jnp.where(evaluable, gamma_a, 1.0)
        zs = conductor.zs * i0_over_i1(safe_gamma_a)
        r_dc_sq = 2 / (a * conductor.sigma)
        return jnp.where(w > 0, zs, r_dc_sq)
