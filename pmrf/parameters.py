"""
Parameters and parameter field specifiers.

Builds on top of `parax`.
"""

import dataclasses
from typing import Any

import equinox as eqx
import parax as prx
from parax.constraints import Interval, Positive as PositiveConstraint
from parax.transforms import Scale
import distreqx.distributions as dist

def _apply_wrappers(
    var: prx.Param, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
    if scale != 1.0:
        var = prx.Derived(Scale(scale), raw_value=var/scale)
    if fixed:
        var = prx.Fixed(var)
    return var

Param = prx.Param

# ---------------------------------------------------------
# Free
# ---------------------------------------------------------

def Free(
    value: prx.Param,
    *,
    scale: float = 1.0,
) -> prx.Param:
    """Create a free parameter."""
    if scale != 1:
        value = _apply_wrappers(value, scale=scale)
    if not prx.is_param(value):
        value = prx.as_param(value)
    return value


def free(
    value: Any = dataclasses.MISSING,
    *,
    scale: float = 1.0,
) -> Any:
    """Create a parameter field specifier that is free by default."""
    if value is dataclasses.MISSING:
        value = 1.0
    def converter(x):
        return Free(x, scale=scale)
    return eqx.field(default=value, converter=converter)

# ---------------------------------------------------------
# Fixed
# ---------------------------------------------------------

def Fixed(
    value: prx.Param,
    *,
    scale: float = 1.0,
) -> prx.Param:
    """Create a fixed parameter."""
    value = _apply_wrappers(value, scale=scale, fixed=True)
    if not prx.is_param(value):
        value = prx.as_param(value)
    return value


def fixed(
    value: Any = dataclasses.MISSING,
    *,
    scale: float = 1.0,
) -> Any:
    """Create a parameter field specifier that is fixed by default."""
    if value is dataclasses.MISSING:
        value = 1.0
    def converter(x):
        return Fixed(x, scale=scale)
    return eqx.field(default=value, converter=converter)

# ---------------------------------------------------------
# Positive
# ---------------------------------------------------------

def Positive(
    value: prx.Param,
    *,
    scale: float = 1.0,
    fixed: bool = False
) -> prx.Param:
    """Create a parameter constrained to be positive."""
    var = prx.Constrained(constraint=PositiveConstraint(), value=value)
    return _apply_wrappers(var, scale, fixed)

def positive(
    value: Any = dataclasses.MISSING,
    *,
    scale: float = 1.0,
    fixed: bool = False
) -> Any:
    """Field specifier for a positive-constrained parameter."""
    if value is dataclasses.MISSING:
        value = 1.0
    def converter(x):
        return Positive(x, scale=scale, fixed=fixed)
    return eqx.field(default=value, converter=converter)

# ---------------------------------------------------------
# Bounded
# ---------------------------------------------------------

def Bounded(
    lower: float, 
    upper: float, 
    *,
    value: prx.Param | None = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
    if value is None:
        value = (lower + upper) / 2.0
    var = prx.Constrained(constraint=Interval(lower, upper), value=value)
    return _apply_wrappers(var, scale, fixed)

def bounded(
    lower: float,
    upper: float,
    *,
    value: Any = dataclasses.MISSING,
    scale: float = 1.0,
    fixed: bool = False
) -> Any:
    if value is dataclasses.MISSING:
        value = (lower + upper) / 2.0
    def converter(x):
        return Bounded(lower, upper, value=x, scale=scale, fixed=fixed)
    return eqx.field(default=value, converter=converter)

# ---------------------------------------------------------
# Uniform
# ---------------------------------------------------------

def Uniform(
    lower: float, 
    upper: float, 
    *,
    value: prx.Param | None = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
    if value is None:
        value = (lower + upper) / 2.0
    var = prx.Random(dist.Uniform(lower, upper), value=value)
    return _apply_wrappers(var, scale, fixed)

def uniform(
    lower: float,
    upper: float,
    *,
    value: Any = dataclasses.MISSING,
    scale: float = 1.0,
    fixed: bool = False
) -> Any:
    if value is dataclasses.MISSING:
        value = (lower + upper) / 2.0
    def converter(x):
        return Uniform(lower, upper, value=x, scale=scale, fixed=fixed)
    return eqx.field(default=value, converter=converter)

# ---------------------------------------------------------
# Normal
# ---------------------------------------------------------

def Normal(
    mean: float, 
    std: float, 
    *,
    value: prx.Param | None = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
    if value is None:
        value = mean
    lower = mean - 2.0 * std
    upper = mean + 2.0 * std
    constrained_var = prx.Constrained(value=value, constraint=Interval(lower, upper))
    var = prx.Random(distribution=dist.Normal(mean, std), raw_value=constrained_var)
    return _apply_wrappers(var, scale, fixed)

def normal(
    mean: float,
    std: float,
    *,
    value: Any = dataclasses.MISSING,
    scale: float = 1.0,
    fixed: bool = False
) -> Any:
    if value is dataclasses.MISSING:
        value = mean
    def converter(x):
        return Normal(mean, std, value=x, scale=scale, fixed=fixed)
    return eqx.field(default=value, converter=converter)

# ---------------------------------------------------------
# CenteredUniform
# ---------------------------------------------------------

def CenteredUniform(
    center: float, 
    half_width: float, 
    *,
    value: prx.Param | None = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
    if value is None:
        value = center
    lower = center - half_width
    upper = center + half_width
    return Uniform(lower, upper, value=value, scale=scale, fixed=fixed)

def centered_uniform(
    center: float,
    half_width: float,
    *,
    value: Any = dataclasses.MISSING,
    scale: float = 1.0,
    fixed: bool = False
) -> Any:
    if value is dataclasses.MISSING:
        value = center
    def converter(x):
        return CenteredUniform(center, half_width, value=x, scale=scale, fixed=fixed)
    return eqx.field(default=value, converter=converter)

# ---------------------------------------------------------
# RelativeNormal
# ---------------------------------------------------------

def RelativeNormal(
    mean: float, 
    pct_std: float, 
    *,
    value: prx.Param | None = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
    if value is None:
        value = mean
    std = mean * pct_std
    return Normal(mean, std, value=value, scale=scale, fixed=fixed)

def relative_normal(
    mean: float,
    pct_std: float,
    *,
    value: Any = dataclasses.MISSING,
    scale: float = 1.0,
    fixed: bool = False
) -> Any:
    if value is dataclasses.MISSING:
        value = mean
    def converter(x):
        RelativeNormal(mean, pct_std, value=x, scale=scale, fixed=fixed)
    return eqx.field(default=value, converter=converter)