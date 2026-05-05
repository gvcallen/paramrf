"""
Parameters and parameter field specifiers.

Builds on top of `parax`.
"""

import dataclasses
from typing import Any

import equinox as eqx
import parax as prx
from parax.constraints import Interval, Positive as PositiveConstraint
from distreqx.distributions import Normal as DistNormal, Uniform as DistUniform

Parameter = prx.ParamLike

from parax import Param as Param

def param(
    default: Parameter = dataclasses.MISSING,
    metadata: dict | None = None,
) -> Any:
    return prx.param(default=default, metadata=metadata)

def _apply_wrappers(
    var: prx.ParamLike, 
    scale: float | str = 1.0, 
    fixed: bool = False
) -> prx.ParamLike:
    if scale != 1.0:
        var = prx.Physical(raw_value=var, scale=scale)
    if fixed:
        var = prx.Fixed(raw_value=var)
    return var

# ---------------------------------------------------------
# Positive
# ---------------------------------------------------------

def Positive(
    value: prx.ParamLike | None = None,
    scale: float | str = 1.0,
    fixed: bool = False
) -> prx.ParamLike:
    """Create a parameter constrained to be positive."""
    if value is None:
        value = 1.0
    var = prx.Constrained(value=value, constraint=PositiveConstraint())
    return _apply_wrappers(var, scale, fixed)

def positive(
    default: Any = dataclasses.MISSING,
    scale: float | str = 1.0,
    fixed: bool = False
) -> Any:
    """Field specifier for a positive-constrained parameter."""
    if default is dataclasses.MISSING:
        default = 1.0
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else Positive(x, scale, fixed)
    return eqx.field(default=default, converter=converter)

# ---------------------------------------------------------
# Bounded
# ---------------------------------------------------------

def Bounded(
    lower: float, 
    upper: float, 
    value: prx.ParamLike | None = None, 
    scale: float | str = 1.0, 
    fixed: bool = False
) -> prx.ParamLike:
    if value is None:
        value = (lower + upper) / 2.0
    var = prx.Constrained(value=value, constraint=Interval(lower, upper))
    return _apply_wrappers(var, scale, fixed)

def bounded(
    lower: float,
    upper: float,
    default: Any = dataclasses.MISSING,
    scale: float | str = 1.0,
    fixed: bool = False
) -> Any:
    if default is dataclasses.MISSING:
        default = (lower + upper) / 2.0
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else Bounded(lower, upper, x, scale, fixed)
    return eqx.field(default=default, converter=converter)

# ---------------------------------------------------------
# Uniform
# ---------------------------------------------------------

def Uniform(
    lower: float, 
    upper: float, 
    value: prx.ParamLike | None = None, 
    scale: float | str = 1.0, 
    fixed: bool = False
) -> prx.ParamLike:
    if value is None:
        value = (lower + upper) / 2.0
    constrained_var = prx.Constrained(value=value, constraint=Interval(lower, upper))
    var = prx.Random(distribution=DistUniform(lower, upper), raw_value=constrained_var)
    return _apply_wrappers(var, scale, fixed)

def uniform(
    lower: float,
    upper: float,
    default: Any = dataclasses.MISSING,
    scale: float | str = 1.0,
    fixed: bool = False
) -> Any:
    if default is dataclasses.MISSING:
        default = (lower + upper) / 2.0
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else Uniform(lower, upper, x, scale, fixed)
    return eqx.field(default=default, converter=converter)

# ---------------------------------------------------------
# Normal
# ---------------------------------------------------------

def Normal(
    mean: float, 
    std: float, 
    value: prx.ParamLike | None = None, 
    scale: float | str = 1.0, 
    fixed: bool = False
) -> prx.ParamLike:
    if value is None:
        value = mean
    lower = mean - 2.0 * std
    upper = mean + 2.0 * std
    constrained_var = prx.Constrained(value=value, constraint=Interval(lower, upper))
    var = prx.Random(distribution=DistNormal(mean, std), raw_value=constrained_var)
    return _apply_wrappers(var, scale, fixed)

def normal(
    mean: float,
    std: float,
    default: Any = dataclasses.MISSING,
    scale: float | str = 1.0,
    fixed: bool = False
) -> Any:
    if default is dataclasses.MISSING:
        default = mean
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else Normal(mean, std, x, scale, fixed)
    return eqx.field(default=default, converter=converter)

# ---------------------------------------------------------
# CenteredUniform
# ---------------------------------------------------------

def CenteredUniform(
    center: float, 
    half_width: float, 
    value: prx.ParamLike | None = None, 
    scale: float | str = 1.0, 
    fixed: bool = False
) -> prx.ParamLike:
    if value is None:
        value = center
    lower = center - half_width
    upper = center + half_width
    return Uniform(lower, upper, value=value, scale=scale, fixed=fixed)

def centered_uniform(
    center: float,
    half_width: float,
    default: Any = dataclasses.MISSING,
    scale: float | str = 1.0,
    fixed: bool = False
) -> Any:
    if default is dataclasses.MISSING:
        default = center
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else CenteredUniform(center, half_width, x, scale, fixed)
    return eqx.field(default=default, converter=converter)

# ---------------------------------------------------------
# RelativeNormal
# ---------------------------------------------------------

def RelativeNormal(
    mean: float, 
    pct_std: float, 
    value: prx.ParamLike | None = None, 
    scale: float | str = 1.0, 
    fixed: bool = False
) -> prx.ParamLike:
    if value is None:
        value = mean
    std = mean * pct_std
    return Normal(mean, std, value=value, scale=scale, fixed=fixed)

def relative_normal(
    mean: float,
    pct_std: float,
    default: Any = dataclasses.MISSING,
    scale: float | str = 1.0,
    fixed: bool = False
) -> Any:
    if default is dataclasses.MISSING:
        default = mean
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else RelativeNormal(mean, pct_std, x, scale, fixed)
    return eqx.field(default=default, converter=converter)