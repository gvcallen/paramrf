"""Interfaces for conductor surface-roughness corrections."""
from abc import abstractmethod

import jax.numpy as jnp
from scipy.constants import mu_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.parameters import Param, param


class AbstractRoughness(Module):
    """Abstract base class for a surface-roughness correction."""

    @abstractmethod
    def factor(self, freq: Frequency, sigma, mu_r) -> jnp.ndarray:
        """Return the multiplier applied to a smooth surface impedance."""


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
    limited -- :class:`~pmrf.materials.conductor.RoughConductor` therefore
    extends :class:`~pmrf.materials.conductor.BulkConductor` rather than the
    abstract conductor, and a non-bulk conductor cannot silently acquire one.
    The arctangent fit is empirical and saturates at $2$: it understates loss
    once $\Delta/\delta$ exceeds roughly unity, which is the documented
    limitation of the Hammerstad-Jensen form rather than a rejected input.

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
