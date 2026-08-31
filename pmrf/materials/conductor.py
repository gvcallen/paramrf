"""
Conductor materials.

A conductor owns the series loss of a line: its surface impedance in ohm per
square, and the bulk conductivity that geometry-aware formulations need
directly.
"""
from __future__ import annotations

from abc import abstractmethod

import jax.numpy as jnp
from scipy.constants import mu_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.parameters import Param, param
from pmrf.utils import field


class AbstractConductor(Module):
    r"""Abstract base class for a conductor material.

    The material interfaces are geometry-free. Geometry-aware formulations
    receive these evaluated quantities together with their geometry.
    """

    @abstractmethod
    def surface_impedance(self, freq: Frequency) -> jnp.ndarray:
        """Complex surface impedance in ohm per square, shape ``(freq.npoints,)``."""

    @abstractmethod
    def sigma(self, freq: Frequency) -> jnp.ndarray:
        """Bulk conductivity in S/m, for formulations that need it directly."""

    @abstractmethod
    def mu_r(self, freq: Frequency) -> jnp.ndarray:
        r"""Relative permeability, shape ``(freq.npoints,)``.

        Relative rather than absolute, so that it composes with
        :meth:`AbstractDielectric.mu_r` and with the relative permittivity
        without a caller having to track which convention a material used.

        The return may be complex, following the same passive convention as the
        permittivity, $\mu_r = \mu' - j\mu''$ with $\mu'' \geq 0$. A material
        with magnetic loss is expressed that way rather than through a separate
        loss field.
        """

    @abstractmethod
    def skin_depth(self, freq: Frequency) -> jnp.ndarray:
        """Skin depth in meters."""


class AbstractRoughness(Module):
    """Abstract base class for a surface-roughness correction."""

    @abstractmethod
    def factor(self, skin_depth: jnp.ndarray) -> jnp.ndarray:
        """The multiplicative correction applied to a smooth surface impedance."""


class Hammerstad(AbstractRoughness):
    r"""
    Hammerstad-Jensen surface-roughness correction.

    **Mathematical Formulation**

    $$K(\omega) = 1 + \frac{2}{\pi}\arctan\left(1.4\left(\frac{\Delta}{\delta}\right)^2\right)$$

    where $\Delta$ is the RMS surface roughness and $\delta$ the skin depth. The
    factor saturates at $2$, which is the well-known limitation of the model.

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

    def factor(self, skin_depth: jnp.ndarray) -> jnp.ndarray:
        ratio = self.roughness / skin_depth
        return 1 + (2 / jnp.pi) * jnp.arctan(1.4 * ratio**2)


class BulkConductor(AbstractConductor):
    r"""
    Smooth bulk metal, in the strong skin-effect regime.

    **Mathematical Formulation**

    $$Z_s(\omega) = \sqrt{\frac{j\omega\mu}{\sigma}}
    = \sqrt{\frac{\omega\mu\rho}{2}}\,(1 + j)
    \qquad
    \delta = \sqrt{\frac{2\rho}{\omega\mu}}$$

    where $\mu = \mu_0\mu_r$ and $\sigma = 1/\rho$. The real part is the skin
    resistance per square and the imaginary part is the internal reactance,
    equal to it in this regime.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.materials import BulkConductor

        copper = BulkConductor(rho=1.68e-8)
        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        zs = copper.surface_impedance(freq)

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 1.7. Wiley.

    Parameters
    ----------
    rho : Param, default=1.68e-8
        Resistivity in ohm-meters. Defaults to copper.
    mu_rel : Param, default=1.0
        Relative permeability of the conductor. Stored under `mu_rel` because
        :meth:`mu_r` is the evaluated accessor; a dataclass field of that name
        would shadow it.
    """
    #: Resistivity in ohm-meters
    rho: Param = param(default=1.68e-8, constraint=Positive())

    #: Relative permeability of the conductor
    mu_rel: Param = param(default=1.0, constraint=Positive())

    @classmethod
    def from_sigma(cls, sigma, **kwargs):
        """Build a conductor from a conductivity in S/m instead of a resistivity."""
        return cls(rho=1.0 / sigma, **kwargs)

    def sigma(self, freq: Frequency) -> jnp.ndarray:
        return (1.0 / self.rho) * jnp.ones(freq.npoints)

    def mu_r(self, freq: Frequency) -> jnp.ndarray:
        return self.mu_rel * jnp.ones(freq.npoints)

    def skin_depth(self, freq: Frequency) -> jnp.ndarray:
        # Guard DC, where the skin depth is infinite, using the double-`where`
        # pattern so the gradient stays finite as well as the value.
        w = jnp.asarray(freq.w)
        safe_w = jnp.where(w > 0, w, 1.0)
        depth = jnp.sqrt(2 * self.rho / (safe_w * mu_0 * self.mu_rel))
        return jnp.where(w > 0, depth, jnp.inf)

    def surface_impedance(self, freq: Frequency) -> jnp.ndarray:
        # sqrt is a branch point at w = 0, so guard it the same way: the
        # impedance is zero there, but its raw gradient would not be.
        w = jnp.asarray(freq.w)
        safe_w = jnp.where(w > 0, w, 1.0)
        rs = jnp.where(w > 0, jnp.sqrt(safe_w * mu_0 * self.mu_rel * self.rho / 2), 0.0)
        return rs * (1 + 1j)


class RoughConductor(BulkConductor):
    r"""
    Bulk metal with a surface-roughness correction.

    **Mathematical Formulation**

    $$Z_s(\omega) = K(\omega) \sqrt{\frac{j\omega\mu}{\sigma}}$$

    where $K$ is supplied by the `roughness` formulation, and equals $1$ for a
    perfectly smooth surface. See :class:`Hammerstad` for the default.

    References
    ----------
    Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
    Computer-Aided Design. IEEE MTT-S International Microwave Symposium Digest,
    407-409.

    Parameters
    ----------
    rho : Param, default=1.68e-8
        Resistivity in ohm-meters.
    mu_rel : Param, default=1.0
        Relative permeability of the conductor.
    roughness : AbstractRoughness, default=Hammerstad()
        The roughness correction formulation. A scalar RMS roughness in meters
        is coerced into a :class:`Hammerstad` correction.
    """
    #: The roughness correction formulation
    roughness: AbstractRoughness = field(
        default_factory=Hammerstad,
        converter=lambda x: x if isinstance(x, AbstractRoughness) else Hammerstad(x),
    )

    def surface_impedance(self, freq: Frequency) -> jnp.ndarray:
        factor = self.roughness.factor(self.skin_depth(freq))
        return super().surface_impedance(freq) * factor


def as_conductor(value) -> AbstractConductor:
    """
    Coerce a value into a conductor material.

    Accepts an existing :class:`AbstractConductor` or a scalar resistivity in
    ohm-meters, which builds a :class:`BulkConductor`.

    Parameters
    ----------
    value : Any
        The value to coerce.

    Returns
    -------
    AbstractConductor
        The resulting conductor material.
    """
    if isinstance(value, AbstractConductor):
        return value
    return BulkConductor(rho=value)
