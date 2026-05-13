"""
Probability distributions classes and factories.

Can be used for random parameters in :mod:`pmrf.parameters`.

Builds on top of the library `distreqx <https://github.com/lockwo/distreqx>`_.
"""
from typing import Any

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
) -> AbstractDistribution:
    """
    Create a Normal distribution defined by relative standard deviation.
    """
    mean, pct_std = jnp.asarray(mean), jnp.asarray(pct_std)

    std = mean * pct_std
    return Normal(mean, std)


def RelativeTruncatedNormal(
    mean: ArrayLike,
    pct_std: ArrayLike,
    trunc_std: ArrayLike = 3.0,
) -> AbstractDistribution:
    """
    Create a symmetric Truncated Normal distribution defined by relative standard deviation
    and truncated at a specified number of standard deviations from the mean.
    """
    from distreqx.distributions import TruncatedNormal 
    
    mean, pct_std, trunc_std = (
        jnp.asarray(mean), 
        jnp.asarray(pct_std), 
        jnp.asarray(trunc_std)
    )

    std = mean * pct_std
    low = mean - (trunc_std * std)
    high = mean + (trunc_std * std)
    
    return TruncatedNormal(loc=mean, scale=std, low=low, high=high)


def truncate_distribution(
    dist: AbstractDistribution, 
    new_lower: Any, 
    new_upper: Any
) -> AbstractDistribution:
    """
    Attempts to return a truncated version of the distribution.
    Raise an exception if unavailable.
    """
    if isinstance(dist, Normal):
        try:
            from distreqx.distributions import TruncatedNormal
            return TruncatedNormal(loc=dist.loc, scale=dist.scale, low=new_lower, high=new_upper)
        except ImportError:
            return None
    elif isinstance(dist, Uniform):
        return Uniform(new_lower, new_upper)
    else:
        raise ValueError(f"Distribution of type {type(dist)} cannot be truncated")
    

__all__ = [
    "AbstractDistribution",
    "Normal",
    "Uniform",
    "Gamma",
    "CenteredUniform",
    "RelativeNormal",
    "RelativeTruncatedNormal",
]

try:
    from distreqx.distributions import LogNormal as LogNormal
    __all__.append('LogNormal')
except:
    pass

try:
    from distreqx.distributions import TruncatedNormal as TruncatedNormal
    __all__.append('TruncatedNormal')
except:
    pass