"""
Conductor materials.

A conductor owns the series loss of a line: its surface impedance in ohm per
square, and the bulk conductivity that geometry-aware formulations need
directly.
"""
from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import mu_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.parameters import Param, param
from pmrf.materials.properties import ConductorProperties
from pmrf.utils import field


class AbstractConductor(Module):
    r"""Abstract base class for a conductor material.

    The material interfaces are geometry-free. Geometry-aware formulations
    receive these evaluated quantities together with their geometry.
    """

    @abstractmethod
    def properties(self, freq: Frequency) -> ConductorProperties:
        """Evaluate surface impedance, conductivity, and permeability."""


class AbstractRoughness(Module):
    """Abstract base class for a surface-roughness correction."""

    @abstractmethod
    def factor(self, freq: Frequency, sigma, mu_r) -> jnp.ndarray:
        """The multiplicative correction applied to a smooth surface impedance."""


class HammerstadRoughness(AbstractRoughness):
    r"""
    Hammerstad-Jensen surface-roughness correction.

    **Mathematical Formulation**

    $$K(\omega) = 1 + \frac{2}{\pi}\arctan\left(1.4\left(\frac{\Delta}{\delta}\right)^2\right)$$

    where $\Delta$ is the RMS surface roughness and $\delta$ the skin depth. The
    factor saturates at $2$, which is the well-known limitation of the model.

    **Validity**

    A Roughness is a correction to conductor behaviour, not a line formulation:
    it scales a smooth surface impedance and produces no state of its own. The
    skin depth is derived here from frequency, conductivity and permeability
    rather than being part of the conductor interface, so the correction is
    meaningful only for a conductor whose loss is genuinely skin-effect
    limited -- :class:`RoughConductor` therefore extends :class:`BulkConductor`
    rather than the abstract conductor, and a non-bulk conductor cannot silently
    acquire one. The arctangent fit is empirical and saturates at $2$: it
    understates loss once $\Delta/\delta$ exceeds roughly unity, which is the
    documented limitation of the Hammerstad-Jensen form rather than a rejected
    input.

    References
    ----------
    Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
    Computer-Aided Design. IEEE MTT-S International Microwave Symposium Digest,
    407-409.

    Parameters
    ----------
    roughness : Param, default=0.0
        RMS surface roughness in meters.
    """
    #: RMS surface roughness in meters
    roughness: Param = param(default=0.0, constraint=Positive())

    def factor(self, freq: Frequency, sigma, mu_r) -> jnp.ndarray:
        w = jnp.asarray(freq.w)
        safe_w = jnp.where(w > 0, w, 1.0)
        skin_depth = jnp.where(
            w > 0, jnp.sqrt(2 / (safe_w * mu_0 * mu_r * sigma)), jnp.inf
        )
        ratio = self.roughness / skin_depth
        return 1 + (2 / jnp.pi) * jnp.arctan(1.4 * ratio**2)


class BulkConductor(AbstractConductor):
    r"""
    Smooth bulk metal, in the strong skin-effect regime.

    **Mathematical Formulation**

    $$Z_s(\omega) = \sqrt{\frac{j\omega\mu}{\sigma}}
    \qquad
    \delta = \sqrt{\frac{2}{\omega\mu\sigma}}$$

    where $\mu = \mu_0\mu_r$. The real part is the skin resistance per square
    and the imaginary part is the internal reactance, equal to it in this
    regime.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.materials import BulkConductor

        copper = BulkConductor(sigma=5.8e7)
        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        zs = copper.properties(freq).zs

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 1.7. Wiley.

    Parameters
    ----------
    sigma : Param, default=5.8e7
        Conductivity in S/m. Defaults to copper.
    mu_r : Param, default=1.0
        Relative permeability of the conductor.
    """
    #: Conductivity in S/m
    sigma: Param = param(default=5.8e7, constraint=Positive())

    #: Relative permeability of the conductor
    mu_r: Param = param(default=1.0, constraint=Positive())

    @classmethod
    def from_rho(cls, rho, **kwargs):
        """Build a conductor from a resistivity in ohm-meters instead of a conductivity."""
        return cls(sigma=1.0 / rho, **kwargs)

    def properties(self, freq: Frequency) -> ConductorProperties:
        # sqrt is a branch point at w = 0, so guard it the same way: the
        # impedance is zero there, but its raw gradient would not be.
        w = jnp.asarray(freq.w)
        safe_w = jnp.where(w > 0, w, 1.0)
        rs = jnp.where(w > 0, jnp.sqrt(safe_w * mu_0 * self.mu_r / (2 * self.sigma)), 0.0)
        ones = jnp.ones(freq.npoints)
        return ConductorProperties(rs * (1 + 1j), self.sigma * ones, self.mu_r * ones)


class RoughConductor(BulkConductor):
    r"""
    Bulk metal with a surface-roughness correction.

    **Mathematical Formulation**

    $$Z_s(\omega) = K(\omega) \sqrt{\frac{j\omega\mu}{\sigma}}$$

    where $K$ is supplied by the `roughness` formulation, and equals $1$ for a
    perfectly smooth surface. See :class:`HammerstadRoughness` for the default.

    References
    ----------
    Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
    Computer-Aided Design. IEEE MTT-S International Microwave Symposium Digest,
    407-409.

    Parameters
    ----------
    sigma : Param, default=5.8e7
        Conductivity in S/m.
    mu_r : Param, default=1.0
        Relative permeability of the conductor.
    roughness : AbstractRoughness, default=HammerstadRoughness()
        The roughness correction formulation. A scalar RMS roughness in meters
        is coerced into a :class:`HammerstadRoughness` correction.
    """
    #: The roughness correction formulation
    roughness: AbstractRoughness = field(
        default_factory=HammerstadRoughness,
        converter=lambda x: x if isinstance(x, AbstractRoughness) else HammerstadRoughness(x),
    )

    def properties(self, freq: Frequency) -> ConductorProperties:
        properties = super().properties(freq)
        factor = self.roughness.factor(freq, properties.sigma, properties.mu_r)
        return eqx.tree_at(lambda p: p.zs, properties, properties.zs * factor)


def as_conductor(value) -> AbstractConductor:
    """
    Coerce a value into a conductor material.

    Accepts an existing :class:`AbstractConductor` or a scalar conductivity in
    S/m, which builds a :class:`BulkConductor`.

    Parameters
    ----------
    value : Any
        The value to coerce.

    Returns
    -------
    AbstractConductor
        The resulting conductor material.

    Raises
    ------
    ValueError
        If ``value`` falls in the resistivity regime (roughly 1e-8 to 1e-5
        ohm-meters for metals) rather than the conductivity regime (roughly
        1e5 to 1e8 S/m): the two are fifteen orders of magnitude apart, so
        there is no ambiguous middle ground.
    """
    if isinstance(value, AbstractConductor):
        return value
    if 0 < value < 1.0:
        raise ValueError(
            f"{value!r} looks like a resistivity in ohm-meters, not a "
            "conductivity in S/m; use BulkConductor.from_rho() instead"
        )
    return BulkConductor(sigma=value)
