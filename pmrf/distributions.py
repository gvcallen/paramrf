"""
Distributions for parameters.

Re-exports of `distreqx.distributions`.
"""
import jax.numpy as jnp
from jaxtyping import ArrayLike

from distreqx.distributions import (
    Normal as Normal,
    LogNormal as LogNormal,
    Uniform as Uniform,
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
    "Normal",
    "LogNormal",
    "Uniform",
    "CenteredUniform",
    "RelativeNormal",
]