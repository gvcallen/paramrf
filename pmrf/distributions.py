"""
Probability distributions and factories.

Can be used for random parameters in :mod:`pmrf.parameters`.

Builds on top of the library `distreqx <https://github.com/lockwo/distreqx>`_.
"""
import jax.numpy as jnp
from jaxtyping import ArrayLike

from distreqx.distributions import (
    AbstractDistribution as AbstractDistribution,
    Normal as Normal,
    Uniform as Uniform,
    Gamma as Gamma,
)


def CenteredUniform(
    center: ArrayLike, 
    half_width: ArrayLike, 
) -> Uniform:
    """
    Create a Uniform distribution defined by center and width.
    """
    center, half_width = jnp.asarray(center), jnp.asarray(half_width)

    lower = center - half_width
    upper = center + half_width
    return Uniform(lower, upper)


def RelativeNormal(
    mean: ArrayLike, 
    pct_std: ArrayLike, 
) -> Normal:
    """
    Create a Normal distribution defined by relative standard deviation.
    """
    mean, pct_std = jnp.asarray(mean), jnp.asarray(pct_std)

    std = mean * pct_std
    return Normal(mean, std)


__all__ = [
    "AbstractDistribution",
    "Normal",
    "Uniform",
    "Gamma",
    "CenteredUniform",
    "RelativeNormal",
]

try:
    from distreqx.distributions import LogNormal as LogNormal
    __all__.extend('LogNormal')
except:
    pass