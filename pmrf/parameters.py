from typing import Sequence
import warnings

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from pmrf.core.parameter import Parameter
from pmrf.utils import stack_vectorized_distributions

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
    
    return Normal(mean=mean_arr, std=sigma, **kwargs)

def Fixed(value, n: int | None = None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` that is marked as fixed.
    
    This sets the `fixed` flag of the parameter to `True`,
    and assigned an infinitely wide improper normal distribution
    if a distribution is not passed.

    Parameters
    ----------
    value
        The value of the parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created fixed Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        value = jnp.array(value)
        
    if not 'distribution' in kwargs:
        dists = dist.ImproperUniform(
            constraints.real, 
            batch_shape=value.shape, 
            event_shape=()
        )
        kwargs['distribution'] = dists        
        
    return Parameter(value=value, fixed=True, **kwargs)

def Free(value, n: int | None = None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` that is marked as not free (i.e., free to vary).
    
    This sets the `fixed` flag of the parameter to `False`,
    and assigned an infinitely wide improper normal distribution
    if a distribution is not passed.

    Parameters
    ----------
    value
        The value of the parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created free Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        value = jnp.array(value)
        
    if not 'distribution' in kwargs:
        dists = dist.ImproperUniform(
            constraints.real, 
            batch_shape=value.shape, 
            event_shape=()
        )
        kwargs['distribution'] = dists
        
    return Parameter(value=value, **kwargs)

def Stacked(parameters: Sequence[Parameter], name: str | None = None, **kwargs) -> Parameter:
    """
    Combine multiple scalar or identically-shaped Parameters into a single vectorized Parameter.
    
    This acts as the inverse of `Parameter.flattened()`.

    Parameters
    ----------
    parameters : Sequence[Parameter]
        The list/tuple of Parameter objects to stack.
    name : str, optional
        The overarching name for the new stacked parameter.
    **kwargs
        Additional arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        A new parameter containing the stacked values and distributions.
    """
    if not parameters:
        raise ValueError("Cannot stack an empty sequence of parameters.")
        
    # 1. Stack the unscaled values
    values = jnp.stack([p.value for p in parameters])
    
    # 2. Combine distributions
    dists = [p.distribution for p in parameters]
    stacked_dist = stack_vectorized_distributions(dists)
    
    # 3. Preserve or generate flat names
    flat_names = []
    for p in parameters:
        if p.flat_names is not None:
            flat_names.extend(p.flat_names)
        elif p.name is not None:
            flat_names.append(p.name)
        else:
            flat_names.append(None)
            
    # 4. Handle the 'fixed' flag
    # Note: Parameter.size evaluates `if self.fixed:`, meaning `fixed` must remain a scalar bool.
    fixed_flags = [p.fixed for p in parameters]
    if not all(f == fixed_flags[0] for f in fixed_flags):
        raise ValueError(
            "All parameters must have the exact same 'fixed' status to be stacked. "
            "Element-wise fixed arrays are not supported by the base Parameter class."
        )
        
    # 5. Handle scales (we DON'T allow heterogeneous scales)
    scales = [p.scale for p in parameters]
    if not all(s == scales[0] for s in scales):
        raise Exception("Cannot create a stacked Parameter with differing scales")
    
    scale = scales[0]
    return Parameter(
        value=values,
        distribution=stacked_dist,
        fixed=fixed_flags[0],
        scale=scale,
        name=name,
        flat_names=flat_names,
        **kwargs
    )