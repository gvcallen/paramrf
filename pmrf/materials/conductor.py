"""
Conductor materials.

A conductor owns the series loss of a line: its surface impedance in ohm per
square, and the bulk conductivity that geometry-aware formulations need
directly.
"""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from scipy.constants import mu_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.parameters import Param, param
from pmrf.utils import field


class AbstractConductor(Module):
    r"""Abstract base class for a conductor material.

    The surface impedance interface is deliberately geometry-free, in ohm per
    square, which is the right currency for planar lines. Coaxial lines are the
    exception: the Schelkunoff solution needs the conductor radius inside a
    Bessel function ratio, and Tesche's equivalent circuit needs the internal
    inductance of a specific rod or tube. Those formulations consume
    :meth:`sigma` plus the radius directly, and :meth:`surface_impedance`
    covers planar lines and the coaxial high-frequency limit.
    """

    def surface_impedance(self, freq: Frequency) -> jnp.ndarray:
        """Complex surface impedance in ohm per square, shape ``(freq.npoints,)``."""
        raise NotImplementedError

    def sigma(self, freq: Frequency) -> jnp.ndarray:
        """Bulk conductivity in S/m, for formulations that need it directly."""
        raise NotImplementedError

    def skin_depth(self, freq: Frequency) -> jnp.ndarray:
        r"""Skin depth $\delta = \sqrt{2\rho / (\omega\mu)}$ in meters."""
        raise NotImplementedError


class Bulk(AbstractConductor):
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
        from pmrf.materials import Bulk

        copper = Bulk(rho=1.68e-8)
        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        zs = copper.surface_impedance(freq)

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 1.7. Wiley.

    Parameters
    ----------
    rho : Param, default=1.68e-8
        Resistivity in ohm-meters. Defaults to copper.
    mur : Param, default=1.0
        Relative permeability of the conductor.
    """
    #: Resistivity in ohm-meters
    rho: Param = param(default=1.68e-8, constraint=Positive())

    #: Relative permeability of the conductor
    mur: Param = param(default=1.0, constraint=Positive())

    @classmethod
    def from_sigma(cls, sigma, **kwargs) -> "Bulk":
        """Build a :class:`Bulk` conductor from a conductivity in S/m."""
        return cls(rho=1.0 / sigma, **kwargs)

    def sigma(self, freq: Frequency) -> jnp.ndarray:
        return (1.0 / self.rho) * jnp.ones(freq.npoints)

    def skin_depth(self, freq: Frequency) -> jnp.ndarray:
        w = jnp.asarray(freq.w)
        safe_w = jnp.where(w > 0, w, 1.0)
        depth = jnp.sqrt(2 * self.rho / (safe_w * mu_0 * self.mur))
        return jnp.where(w > 0, depth, jnp.inf)

    def surface_impedance(self, freq: Frequency) -> jnp.ndarray:
        rs = jnp.sqrt(freq.w * mu_0 * self.mur * self.rho / 2)
        return rs * (1 + 1j)


class RoughConductor(Bulk):
    r"""
    Bulk metal with a surface-roughness correction.

    **Mathematical Formulation**

    The Hammerstad-Jensen correction scales the smooth surface impedance by

    $$K(\omega) = 1 + \frac{2}{\pi}\arctan\left(1.4\left(\frac{\Delta}{\delta}\right)^2\right)$$

    where $\Delta$ is the RMS surface roughness and $\delta$ the skin depth. The
    factor saturates at $2$, which is the well-known limitation of the model.

    With ``rms_roughness=0`` this is identical to :class:`Bulk`.

    References
    ----------
    Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
    Computer-Aided Design. IEEE MTT-S International Microwave Symposium Digest,
    407-409.

    Parameters
    ----------
    rho : Param, default=1.68e-8
        Resistivity in ohm-meters.
    rms_roughness : Param, default=0.0
        RMS surface roughness in meters.
    mur : Param, default=1.0
        Relative permeability of the conductor.
    model : {'hammerstad'}, default='hammerstad'
        The roughness correction model.
    """
    #: RMS surface roughness in meters
    rms_roughness: Param = param(default=0.0, constraint=Positive())

    #: The roughness correction model
    model: Literal["hammerstad"] = field(default="hammerstad", static=True)

    def __post_init__(self):
        if self.model != "hammerstad":
            raise ValueError(f"Unknown roughness model: {self.model!r}")

    def roughness_factor(self, freq: Frequency) -> jnp.ndarray:
        """The multiplicative correction applied to the smooth surface impedance."""
        ratio = self.rms_roughness / self.skin_depth(freq)
        return 1 + (2 / jnp.pi) * jnp.arctan(1.4 * ratio**2)

    def surface_impedance(self, freq: Frequency) -> jnp.ndarray:
        return super().surface_impedance(freq) * self.roughness_factor(freq)


def as_conductor(value) -> AbstractConductor:
    """
    Coerce a value into a conductor material.

    Accepts an existing :class:`AbstractConductor` or a scalar resistivity in
    ohm-meters, which builds a :class:`Bulk` conductor.

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
    return Bulk(rho=value)
