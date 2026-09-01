"""Current-distribution strategies for conductor loss in planar lines."""
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import epsilon_0, mu_0

from pmrf.frequency import Frequency
from pmrf.materials.conductor_shape import (
    HalfSpaceShape,
    HollowayKuesterSlabShape,
    RootSumSquareSlabShape,
)


def _wheeler_current_factor(zc):
    """Return Wheeler's dimensionless current-crowding factor."""
    z0 = jnp.sqrt(mu_0 / epsilon_0)
    return jnp.exp(-1.2 * (jnp.real(zc) / z0) ** 0.7)


def _slab_penetration(freq, conductor, t):
    """Return the guarded finite-slab penetration transition."""
    evaluable = (freq.w > 0) & jnp.isfinite(conductor.sigma)
    safe_sigma = jnp.where(evaluable, conductor.sigma, 1.0)
    safe_zs = jnp.where(evaluable, conductor.zs, 0.0)
    return jnp.where(
        evaluable, jnp.abs(jnp.tanh(safe_sigma * safe_zs * t / 2)), 0.0,
    )


class AbstractCurrentDistribution(eqx.Module):
    r"""Surface-current distribution for a transmission-line cross-section.

    A strategy returns ``(shape, weight)`` pairs.  The shape supplies the
    material's surface impedance and the weight, in inverse metres, charges
    that impedance into the line's series impedance.  Weights are evaluated
    at the requested frequency because current crowding may be frequency
    dependent.
    """

    @abstractmethod
    def distribute(self, freq: Frequency, *, zc, **geometry):
        r"""Return the conductor-shape and geometry-weight pairs."""
        raise NotImplementedError


class WheelerCurrentDistribution(AbstractCurrentDistribution):
    r"""Wheeler's incremental-inductance current distribution.

    **Mathematical Formulation**

    $$k_c = \frac{2}{W}\exp\left[-1.2\left(\frac{\Re(Z_c)}{Z_0}\right)^{0.7}\right]$$

    The trace is represented by a half-space surface and the returned weight
    is Wheeler's geometry factor in inverse metres.

    References
    ----------
    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.
    """

    def distribute(self, freq: Frequency, *, zc, w, t=None, conductor=None, **geometry):
        weight = 2 / w * _wheeler_current_factor(zc)
        shape = (
            HalfSpaceShape() if t is None else
            RootSumSquareSlabShape(dc_shape_factor=1 / (w * t * weight))
        )
        return ((shape, weight),)


class TraceGroundCurrentDistribution(AbstractCurrentDistribution):
    r"""Independent microstrip trace and ground-plane current distributions.

    **Mathematical Formulation**

    Holloway and Kuester's ground-plane current density for a uniform strip
    current is

    $$\frac{J_g(x)}{I}=\frac{1}{\pi W}\left[
    \tan^{-1}\left(\frac{2x-W}{2h}\right)-
    \tan^{-1}\left(\frac{2x+W}{2h}\right)\right].$$

    Defining the effective ground width by
    $1/W_g=\int_{-\infty}^{\infty}|J_g/I|^2dx$ gives

    $$\frac{1}{W_g}=\frac{2}{\pi W^2}\left[
    W\tan^{-1}\left(\frac{W}{2h}\right)-
    h\log\left(1+\frac{W^2}{4h^2}\right)\right].$$

    The trace uses the finite-thickness total-current slab. Its weight moves
    from $1/(2W)$ at dc, which gives the exact trace resistance
    $1/(\sigma Wt)$, to Wheeler's $2K_i/W$ strong-skin trace weight. The
    transition is $|\tanh(\gamma_ct/2)|$, the same dimensionless penetration
    factor appearing in the slab impedance. The ground is deliberately held
    at its strong-skin half-space form because its dc resistance depends on
    plane width and copper thickness, neither of which microstrip supplies.

    $K_i$ is applied to both trace and ground. Holloway and Kuester derive the
    ground distribution independently of Wheeler's trace-crowding correction,
    so this is a convenient shared edge-crowding convention rather than a
    claim that their ground expression contains $K_i$. Omitting it from the
    ground increases the strong-skin result by about 4--5% for ordinary
    microstrip. Even with it, the split is intentionally not Wheeler's rule:
    it adds a separately integrated ground loss, giving about 14% more loss
    for representative 50-ohm lines.

    References
    ----------
    Holloway, C. L., & Kuester, E. F. (1995). Closed-form expressions for the
    current density on the ground plane of a microstrip line, with application
    to ground plane loss. IEEE Transactions on Microwave Theory and Techniques,
    43(5), 1204-1208. Correction, 54(11), 4018-4019 (2006).

    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.

    Holloway, C. L., & Kuester, E. F. (1994). Edge shape effects and
    quasi-closed form expressions for the conductor loss of microstrip lines.
    Radio Science, 29(3), 539-559.
    """

    def distribute(self, freq: Frequency, *, zc, w, h, t=None, conductor=None):
        ki = _wheeler_current_factor(zc)

        u_over_two = w / (2 * h)
        inverse_ground_width = 2 / (jnp.pi * w**2) * (
            w * jnp.arctan(u_over_two) - h * jnp.log1p(u_over_two**2)
        )
        ground_pair = (HalfSpaceShape(), ki * inverse_ground_width)

        if t is None:
            return ((HalfSpaceShape(), 2 * ki / w), ground_pair)
        if conductor is None:
            raise ValueError("conductor properties are required for finite thickness")

        penetration = _slab_penetration(freq, conductor, t)
        trace_weight = 1 / (2 * w) + penetration * (2 * ki / w - 1 / (2 * w))
        return ((HollowayKuesterSlabShape(t=t), trace_weight), ground_pair)


class CohnCurrentDistribution(AbstractCurrentDistribution):
    r"""Cohn's stripline current distribution.

    **Mathematical Formulation**

    Cohn's $k_c=2\alpha_c Z_c/R_s$ is an effective strong-skin weight for the
    centre strip and both ground planes together.  When the centre-strip
    thickness is known, ParamRF charges that weight through Holloway and
    Kuester's total-current slab and interpolates the weight from $1/(2W)$ at
    dc to $k_c$ in strong skin effect according to the ParamRF convention

    $$k(f)=\frac{1}{2W}+|\tanh(\gamma_cT/2)|
    \left(k_c-\frac{1}{2W}\right).$$

    This gives the centre strip's exact
    $R_{dc}=1/(\sigma WT)$ while preserving Cohn's complete high-frequency
    result and replacing the half-space internal reactance.

    Ground-plane loss is not added as a separate pair.  Unlike microstrip,
    symmetric stripline already has both returns folded into Cohn's effective
    geometry factor; adding ground half-spaces would double-count them.  Their
    finite-thickness dc resistance also cannot be determined without ground
    thickness and width inputs.  A strip with unspecified thickness retains
    the existing zero conductor-loss convention because Cohn's finite-$T$
    expression diverges as $T\to0$.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE Transactions
    on Microwave Theory and Techniques, 3(2), 119-126.

    Holloway, C. L., & Kuester, E. F. (1994). Edge shape effects and
    quasi-closed form expressions for the conductor loss of microstrip lines.
    Radio Science, 29(3), 539-559.

    ParamRF finite-strip interpolation convention; no external source.
    """

    def distribute(self, freq: Frequency, *, zc, w, b, t, ep_r, conductor=None):
        if t is None:
            weight = jnp.asarray(0.0)
        else:
            ep_r = jnp.real(ep_r)
            zc_real = jnp.real(zc)
            a = 1 + 2 * w / (b - t) + (b + t) / (jnp.pi * (b - t)) * jnp.log((2 * b - t) / t)
            alpha_low = 2.7e-3 * ep_r * zc_real / (30 * jnp.pi * (b - t)) * a
            beta = 1 + b / (0.5 * w + 0.7 * t) * (
                0.5 + 0.7 * t / w + jnp.log(4 * jnp.pi * w / t) / (2 * jnp.pi)
            )
            alpha_high = 0.16 / (zc_real * b) * beta
            alpha_over_rs = jnp.where(jnp.sqrt(ep_r) * zc_real < 120, alpha_low, alpha_high)
            skin_weight = 2 * alpha_over_rs * zc_real
            if conductor is None:
                raise ValueError("conductor properties are required for finite thickness")
            penetration = _slab_penetration(freq, conductor, t)
            weight = 1 / (2 * w) + penetration * (skin_weight - 1 / (2 * w))
        shape = HalfSpaceShape() if t is None else HollowayKuesterSlabShape(t=t)
        return ((shape, weight),)
