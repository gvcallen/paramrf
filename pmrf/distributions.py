"""
Probability distribution classes and factories.

Can be used for random parameters in :mod:`pmrf.parameters`.

These are (mostly) re-exports from `parax <https://gvcallen.github.io/parax>`_, which in turn
takes them from the `distreqx <https://lockwo.github.io/distreqx>`_ library, filling in any that
the installed version of `distreqx` does not provide. The goal is to cover the most common
applications; for more advanced use-cases, simply use `distreqx` directly instead.
"""
import jax.numpy as jnp
from jaxtyping import ArrayLike

from parax.distributions import (
    AbstractDistribution as AbstractDistribution,
    Gamma as Gamma,
    Joint as Joint,
    LogNormal as LogNormal,
    Normal as Normal,
    Transformed as Transformed,
    TruncatedNormal as TruncatedNormal,
    Uniform as Uniform,
)

from parax.probability import truncate_distribution as truncate

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
    mean, pct_std, trunc_std = (
        jnp.asarray(mean), 
        jnp.asarray(pct_std), 
        jnp.asarray(trunc_std)
    )

    std = mean * pct_std
    low = mean - (trunc_std * std)
    high = mean + (trunc_std * std)
    
    return TruncatedNormal(loc=mean, scale=std, low=low, high=high)


__all__ = [
    "AbstractDistribution",
    "CenteredUniform",
    "Gamma",
    "Joint",
    "LogNormal",
    "Normal",
    "RelativeNormal",
    "RelativeTruncatedNormal",
    "Transformed",
    "TruncatedNormal",
    "Uniform",
    "truncate",
]
