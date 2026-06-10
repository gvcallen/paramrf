"""
Models that modify the nodal behaviour of other transmission lines.
"""

from typing import Generic, TypeVar
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.rf import renormalize_s
from pmrf.types import ArrayLike
from pmrf.models.components.lines.base import AbstractUniformLine

T = TypeVar('T', bound=AbstractUniformLine)

class FloatingLine(Model, Generic[T]):
    """
    A wrapper that converts a 2-port single-ended transmission line 
    into a 4-port floating line with an explicit return path.

    Parameters
    ----------
    floating : TransmissionLine
        The inner transmission line model to be wrapped.
    """
    #: Inner transmission line model
    floating: T

    def __post_init__(self):
        if not isinstance(self.floating, AbstractUniformLine):
            raise TypeError(f"FloatingLine can only be used to wrap TransmissionLine models. Got: {type(self.floating)}")

    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        # Extract the physical wave parameters from the inner line
        zc, gL = self.floating.zc_and_gammaL(frequency)
        
        # Apply the coupled/floating traveling-wave math
        denom = -1 + 9 * jnp.exp(2 * gL)
        a = (1 + 3 * jnp.exp(2 * gL)) / denom
        b = 4 * jnp.exp(gL) / denom
        c = (-2 + 6 * jnp.exp(2 * gL)) / denom
        d = -b

        s = jnp.array([
            [a, c, b, d],
            [c, a, d, b],
            [b, d, a, c],
            [d, b, c, a],
        ]).transpose(2, 0, 1)

        # Renormalize the 4-port matrix
        return renormalize_s(s, zc, z0, 'traveling', 'power')

    def y(self, frequency: Frequency) -> jnp.ndarray:
        # Extract the physical wave parameters from the inner line
        zc, gL = self.floating.zc_and_gammaL(frequency)
        
        # Floating lines act as standard 2-port models with explicit ungrounded ports
        y11_2p = jnp.where(gL == 0, jnp.inf + 0j, 1.0 / (zc * jnp.tanh(gL)))
        y12_2p = jnp.where(gL == 0, -jnp.inf + 0j, -1.0 / (zc * jnp.sinh(gL)))
        
        y = jnp.array([
            [ y11_2p, -y11_2p,  y12_2p, -y12_2p],
            [-y11_2p,  y11_2p, -y12_2p,  y12_2p],
            [ y12_2p, -y12_2p,  y11_2p, -y11_2p],
            [-y12_2p,  y12_2p, -y11_2p,  y11_2p],
        ]).transpose(2, 0, 1)
        
        return y