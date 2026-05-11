"""
Parameters and parameter field specifiers.

Builds on top of `parax`.
"""

import dataclasses
from typing import Any, Optional, Tuple

import jax.numpy as jnp
import equinox as eqx
import parax as prx
from parax.constraints import Interval, Positive as PositiveConstraint
from parax.transforms import Scale
import distreqx.distributions as dist

Param = prx.Param

def _apply_wrappers(
    var: Param, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Apply scale and fixed transformations to a parameter.

    Parameters
    ----------
    var : Param
        The base parameter to transform.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.

    Returns
    -------
    Param
        The modified parameter wrapped with `Scale` or `Fixed` if necessary.
    """
    if scale != 1.0:
        var = prx.Derived(Scale(scale), raw_value=var/scale)
    if fixed:
        var = prx.Fixed(var)
    return var

# ---------------------------------------------------------
# Parameter Factories
# ---------------------------------------------------------

def Free(
    value: Param,
    *,
    scale: float = 1.0,
    fixed: bool = False,
) -> Param:
    """
    Create a free parameter with optional scaling.

    Parameters
    ----------
    value : Param
        The initial value of the parameter.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.
    fixed : bool, optional
        Whether the parameter should be fixed, by default False.

    Returns
    -------
    Param
        A Parax parameter object.
    """
    if not prx.is_param(value):
        value = prx.as_param(value)
    return _apply_wrappers(value, scale=scale, fixed=fixed)


def Fixed(
    value: Param,
    *,
    scale: float = 1.0,
) -> Param:
    """
    Create a fixed parameter that will not be optimized.

    Parameters
    ----------
    value : Param
        The value to fix the parameter at.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.

    Returns
    -------
    Param
        A fixed Parax parameter object.
    """
    if not prx.is_param(value):
        value = prx.as_param(value)
    return _apply_wrappers(value, scale=scale, fixed=True)


def Positive(
    value: Param,
    *,
    scale: float = 1.0,
    fixed: bool = False
) -> Param:
    """
    Create a parameter constrained to be strictly positive.

    If the input `value` is already constrained with an `Interval`, this function
    will intersect the existing bounds with [0, inf) to preserve the user's 
    original intent while enforcing positivity.

    Parameters
    ----------
    value : Param
        The initial value or an already constrained parameter object.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.
    fixed : bool, optional
        Whether the parameter should be fixed, by default False.

    Returns
    -------
    Param
        A positive-constrained Parax parameter object.
    """
    # Intercept and merge if the value is already an Interval constraint
    if prx.is_bounded(value):
        inner_lower, inner_upper = value.bounds
        new_lower = max(0.0, inner_lower)
        var = prx.Constrained(constraint=Interval(new_lower, inner_upper), value=value)
    else:
        var = prx.Constrained(constraint=PositiveConstraint(), value=value)
        
    return _apply_wrappers(var, scale, fixed)


def Bounded(
    lower: float, 
    upper: float, 
    *,
    value: Optional[Param] = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter constrained within a specific interval.

    Parameters
    ----------
    lower : float
        The lower bound of the interval.
    upper : float
        The upper bound of the interval.
    value : Param, optional
        The initial value. If None, defaults to the midpoint of the bounds.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.
    fixed : bool, optional
        Whether the parameter should be fixed, by default False.

    Returns
    -------
    Param
        An interval-constrained Parax parameter object.
    """
    if prx.is_bounded(value):
        inner_lower, inner_upper = value.bounds
        lower = max(lower, inner_lower)
        upper = max(upper, inner_upper)

    if value is None:
        value = (lower + upper) / 2.0
    var = prx.Constrained(constraint=Interval(lower, upper), value=value)
    return _apply_wrappers(var, scale, fixed)


def Uniform(
    lower: float, 
    upper: float, 
    *,
    value: Optional[Param] = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter initialized from a Uniform distribution.

    Parameters
    ----------
    lower : float
        The lower bound of the uniform distribution.
    upper : float
        The upper bound of the uniform distribution.
    value : Param, optional
        The specific initial value to use instead of a random sample, by default None.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.
    fixed : bool, optional
        Whether the parameter should be fixed, by default False.

    Returns
    -------
    Param
        A Parax Random parameter configured with a Uniform distribution.
    """
    if value is None:
        value = (lower + upper) / 2.0
    var = prx.Random(dist.Uniform(lower, upper), value=value)
    return _apply_wrappers(var, scale, fixed)


def Normal(
    mean: float, 
    std: float, 
    *,
    value: Optional[Param] = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter initialized from a Normal distribution, with fallback bounds.

    The parameter is strictly bounded within ±2 standard deviations from the mean
    to prevent runaway optimization in unconstrained settings.

    Parameters
    ----------
    mean : float
        The mean of the normal distribution.
    std : float
        The standard deviation of the normal distribution.
    value : Param, optional
        The initial value, by default defaults to the mean.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.
    fixed : bool, optional
        Whether the parameter should be fixed, by default False.

    Returns
    -------
    Param
        A Parax Random parameter configured with a Normal distribution and interval constraints.
    """
    if value is None:
        value = mean
    lower = mean - 2.0 * std
    upper = mean + 2.0 * std
    constrained_var = prx.Constrained(value=value, constraint=Interval(lower, upper))
    var = prx.Random(distribution=dist.Normal(mean, std), raw_value=constrained_var)
    return _apply_wrappers(var, scale, fixed)


def CenteredUniform(
    center: float, 
    half_width: float, 
    *,
    value: Optional[Param] = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter initialized from a Uniform distribution defined by center and width.

    Parameters
    ----------
    center : float
        The center point of the uniform distribution.
    half_width : float
        Half of the total width of the distribution (distance from center to bound).
    value : Param, optional
        The specific initial value to use, defaults to the center if None.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.
    fixed : bool, optional
        Whether the parameter should be fixed, by default False.

    Returns
    -------
    Param
        A Parax Random parameter configured with a Uniform distribution.
    """
    if value is None:
        value = center
    lower = center - half_width
    upper = center + half_width
    return Uniform(lower, upper, value=value, scale=scale, fixed=fixed)


def RelativeNormal(
    mean: float, 
    pct_std: float, 
    *,
    value: Optional[Param] = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter initialized from a Normal distribution defined by relative standard deviation.

    Parameters
    ----------
    mean : float
        The mean of the normal distribution.
    pct_std : float
        The standard deviation expressed as a percentage/fraction of the mean (e.g., 0.1 for 10%).
    value : Param, optional
        The initial value, defaults to the mean if None.
    scale : float, optional
        Scaling factor applied to the parameter, by default 1.0.
    fixed : bool, optional
        Whether the parameter should be fixed, by default False.

    Returns
    -------
    Param
        A Parax Random parameter configured with a Normal distribution.
    """
    if value is None:
        value = mean
    std = mean * pct_std
    return Normal(mean, std, value=value, scale=scale, fixed=fixed)


# ---------------------------------------------------------
# Field Specifier
# ---------------------------------------------------------

def param(
    value: Any = dataclasses.MISSING,
    *,
    scale: float = 1.0,
    fixed: bool = False,
    positive: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
) -> Any:
    """
    A field specifier for parameters in models.

    This allows enforcing constraints and scales directly in the model definition.
    For example, a parameter can be given default bounds, enforced to be positive,
    and fixed.

    Parameters
    ----------
    value : Any, optional
        The initial value of the parameter. Defaults to 1.0 if not provided.
    scale : float, optional
        Scaling factor for the parameter, by default 1.0.
    fixed : bool, optional
        If True, the parameter will not be updated during optimization, by default False.
    positive : bool, optional
        If True, enforces a strictly positive constraint on the parameter, by default False.
    bounds : tuple of float, optional
        A tuple of `(lower_bound, upper_bound)` defining an interval constraint. 
        Overrides `positive` if both are specified, though ideally only one should be used. 
        By default None.

    Returns
    -------
    Any
        An `equinox.field` with the appropriate Parax constraint converters applied.
    """
    if value is dataclasses.MISSING:
        value = 1.0

    if prx.is_bounded(value):
        lower, upper = value.bounds
        if bounds is not None:
            lower = max(bounds[0], lower)
            upper = min(bounds[1], upper)
        bounds = lower, upper

    def converter(x):
        x = jnp.array(x)
        if bounds is not None:
            lower, upper = bounds
            if positive:
                lower = max(0.0, lower)
            return Bounded(lower, upper, value=x, scale=scale, fixed=fixed)
        elif positive:
            return Positive(x, scale=scale, fixed=fixed)
        else:
            return Free(x, scale=scale, fixed=fixed)
            
    return eqx.field(default=value, converter=converter)


__all__ = [
    "param",
    "Free",
    "Fixed",
    "Positive",
    "Bounded",
    "Uniform",
    "Normal",
    "CenteredUniform",
    "RelativeNormal",
]
