from typing import Sequence
import warnings

import jax.numpy as jnp
import numpyro.distributions as dist

from pmrf.parameters.parameter import Parameter

def Normal(mean: float | Sequence[float], std: float | Sequence[float], n: int | None = None, value=None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a normal (Gaussian) distribution.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    std : float | Sequence[float]
        The standard deviation of the distribution. Can be a sequence for a multi-valued Parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    value : optional
        The initial value. If None, the mean of the distribution is used. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        mean = jnp.broadcast_to(jnp.array(mean), shape)
        std = jnp.broadcast_to(jnp.array(std), shape)
        if value is not None:
            value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        mean, std = jnp.array(mean), jnp.array(std)
    
    dists = dist.Normal(mean, std)
    values = mean if value is None else value
    return Parameter(value=values, distribution=dists, **kwargs)
    
def PercentNormal(mean: float | Sequence[float], perc: float | Sequence[float], **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a normal (Gaussian) distribution and a percentage standard deviation.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    perc : float | Sequence[float]
        The percentage width to use to initialize the standard deviation,
        assuming the percentage represents +/- 2*sigma (95% coverage).
        As an example, passing `5.0` results in `std = 0.025 * mean`.
        Can be a sequence for a multi-valued Parameter.
    **kwargs
        Additional keyword arguments passed to the `Normal` factory function.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    warnings.warn(
        "PercentNormal is deprecated and will be removed in a future version. "
        "Please use RelativeNormal instead",
        category=DeprecationWarning,
        stacklevel=2
    )        
    
    if isinstance(perc, Sequence) or isinstance(perc, jnp.ndarray):
        std = jnp.array(perc) * jnp.array(mean) / 200.0
    else:
        std = perc * jnp.array(mean) / 200.0
    return Normal(mean=mean, std=std, **kwargs)

def RelativeNormal(mean: float | Sequence[float], std_fraction: float | Sequence[float], **kwargs) -> Parameter:
    r"""
    Create a `Parameter` with a normal distribution defined by a relative standard deviation.

    The scale (sigma) is calculated as: `mean * std_fraction`

    Parameters
    ----------
    mean : float | Sequence[float]
        The center (mean) of the distribution.
    std_fraction : float | Sequence[float]
        The standard deviation expressed as a fraction of the mean 
        (also known as the coefficient of variation).
        e.g., 0.1 results in a distribution with sigma = 0.1 * mean.
    **kwargs
        Additional keyword arguments passed to the `Normal` constructor.

    Returns
    -------
    Parameter
    """
    mean_arr = jnp.array(mean)
    frac_arr = jnp.array(std_fraction)
    
    # Calculate absolute standard deviation
    # sigma = 10% of mean
    sigma = jnp.abs(mean_arr * frac_arr)
    
    return Normal(loc=mean_arr, scale=sigma, **kwargs)