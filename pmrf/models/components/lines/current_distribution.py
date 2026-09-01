"""Current-distribution strategies for conductor loss in planar lines."""
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import epsilon_0, mu_0

from pmrf.frequency import Frequency
from pmrf.materials.conductor_shape import (
    AbstractConductorShape,
    HalfSpaceShape,
    RootSumSquareSlabShape,
)


class AbstractCurrentDistribution(eqx.Module):
    r"""Surface-current distribution for a transmission-line cross-section.

    A strategy returns ``(shape, weight)`` pairs.  The shape supplies the
    material's surface impedance and the weight, in inverse metres, charges
    that impedance into the line's series impedance.  Weights are evaluated
    at the requested frequency because current crowding may be frequency
    dependent.  A strategy chooses shapes and weights only: the
    cross-section dimensions a shape needs reach it from the line at call
    time, so no strategy ever hands a shape a number derived from its own
    weight.
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

    A trace of unspecified thickness is a half-space; a trace of stated
    thickness is charged through :attr:`slab_shape`.

    References
    ----------
    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.
    """

    #: The finite-thickness cross-section entry used when a thickness is
    #: stated.  The default,
    #: :class:`~pmrf.materials.conductor_shape.RootSumSquareSlabShape`, is
    #: the only entry that is right at both asymptotes under this
    #: distribution's frequency-independent weight;
    #: :class:`~pmrf.materials.conductor_shape.HollowayKuesterSlabShape` is
    #: the exact strip-diffusion result but is normalised to the total strip
    #: current and so wants a weight of $1/(2W)$ rather than Wheeler's.
    #: Neither is the better entry in general -- see the normalisation note
    #: on :class:`~pmrf.materials.conductor_shape.AbstractConductorShape`.
    slab_shape: AbstractConductorShape = eqx.field(
        default_factory=RootSumSquareSlabShape
    )

    def distribute(self, freq: Frequency, *, zc, w, t=None, **geometry):
        z0 = jnp.sqrt(mu_0 / epsilon_0)
        weight = 2 / w * jnp.exp(-1.2 * (jnp.real(zc) / z0) ** 0.7)
        shape = HalfSpaceShape() if t is None else self.slab_shape
        return ((shape, weight),)


class CohnCurrentDistribution(AbstractCurrentDistribution):
    r"""Cohn's stripline current distribution.

    **Mathematical Formulation**

    The returned weight is $k_c=2\alpha_c/R_s$, using Cohn's two attenuation
    branches and the stripline geometry.  A zero-thickness strip has no finite
    conductor-loss weight in this model.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE Transactions
    on Microwave Theory and Techniques, 3(2), 119-126.
    """

    def distribute(self, freq: Frequency, *, zc, w, b, t, ep_r, **geometry):
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
            weight = 2 * alpha_over_rs * zc_real
        return ((HalfSpaceShape(), weight),)
