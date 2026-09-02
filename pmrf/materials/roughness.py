"""Interfaces for conductor surface-roughness corrections."""
from abc import abstractmethod

import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.modules.base import Module


class AbstractRoughness(Module):
    """Abstract base class for a surface-roughness correction."""

    @abstractmethod
    def factor(self, freq: Frequency, sigma, mu_r) -> jnp.ndarray:
        """Return the multiplier applied to a smooth surface impedance."""
