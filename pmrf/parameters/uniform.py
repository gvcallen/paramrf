from typing import Sequence
import warnings

import jax.numpy as jnp
import numpyro.distributions as dist

from pmrf.parameters.parameter import Parameter    
    
def Uniform(low: float | Sequence[float], high: float | Sequence[float], value=None, *, n: int | None = None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution.

    Parameters
    ----------
    low : float | Sequence[float]
        The lower value of the distribution. Can be a sequence for a multi-valued Parameter.
    high : float | Sequence[float]
        The upper value of the distribution. Can be a sequence for a multi-valued Parameter.
    value : optional
        The initial value. If None, the midpoint of the distribution is used. Defaults to None.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        low = jnp.broadcast_to(jnp.array(low), shape)
        high = jnp.broadcast_to(jnp.array(high), shape)
        if value is not None:
            value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        low, high = jnp.array(low), jnp.array(high)
    
    dists = dist.Uniform(low, high)
    values = (low + high) / 2.0 if value is None else value
    return Parameter(value=values, distribution=dists, **kwargs)

def PercentUniform(mean: float | Sequence[float], perc: float | Sequence[float], *args, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution defined by a percentage width.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    perc : float | Sequence[float]
        The percentage deviation from the mean to either of the bounds.
        Bounds are calculated as `mean +/- (perc * mean / 200)`.
    **kwargs
        Additional keyword arguments passed to the `Uniform` factory function.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    warnings.warn(
        "PercentUniform is deprecated and will be removed in a future version. "
        "Please use RelativeUniform instead",
        category=DeprecationWarning,
        stacklevel=2
    )    
    
    if isinstance(perc, Sequence) or isinstance(perc, jnp.ndarray):
        delta = jnp.array(perc) * jnp.array(mean) / 200.0
    else:
        delta = perc * jnp.array(mean) / 200.0
    return Uniform(mean-delta, mean+delta, *args, **kwargs)

def RelativeUniform(mean: float | Sequence[float], deviation_fraction: float | Sequence[float], *args, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution defined by a fractional deviation.

    The bounds are calculated as: `mean * (1 +/- deviation_fraction)`

    Parameters
    ----------
    mean : float | Sequence[float]
        The center (mean) of the distribution.
    deviation_fraction : float | Sequence[float]
        The relative radius of the distribution bounds as a fraction of the mean.
        e.g., 0.1 results in bounds of [0.9 * mean, 1.1 * mean].
    **kwargs
        Additional keyword arguments passed to the `Uniform` constructor.

    Returns
    -------
    Parameter
    """
    mean_arr = jnp.array(mean)
    frac_arr = jnp.array(deviation_fraction)
    
    # Calculate the absolute deviation (radius)
    # delta = 10% of mean
    delta = jnp.abs(mean_arr * frac_arr)
    
    return Uniform(mean_arr - delta, mean_arr + delta, *args, **kwargs)

def CenteredUniform(mean: float | Sequence[float], half_width: float | Sequence[float], *args, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a uniform distribution.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean value of the distribution. Can be a sequence for a multi-valued Parameter.
    half_width : float | Sequence[float]
        The half-width value of the distribution. Can be a sequence for a multi-valued Parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    value : optional
        The initial value. If None, the midpoint of the distribution is used. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    low = mean - half_width
    high = mean + half_width
    
    return Uniform(low, high, *args, **kwargs)