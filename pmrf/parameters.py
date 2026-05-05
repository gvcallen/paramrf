"""
Parameters and parameter field specifiers.

Builds on top of `parax`.
"""

import dataclasses
from typing import Any

import jax.numpy as jnp
import equinox as eqx
import parax as prx
from parax.constraints import Interval, Positive as PositiveConstraint
from parax.transforms import Scale
from distreqx.distributions import Normal as DistNormal, Uniform as DistUniform

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
    scale: float = 1.0,
) -> prx.Param:
    """Create a free parameter."""
    if scale != 1:
        value = _apply_wrappers(value, scale=scale)
    if not prx.is_param(value):
        value = prx.as_param(value)
    return value


def free(
    default: Any = dataclasses.MISSING,
    scale: float = 1.0,
) -> Any:
    """Create a parameter field specifier that is free by default."""
    if default is dataclasses.MISSING:
        default = 1.0
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else Free(x, scale)
    return eqx.field(default=default, converter=converter)

# ---------------------------------------------------------
# Fixed
# ---------------------------------------------------------

def Fixed(
    value: prx.Param,
    scale: float = 1.0,
) -> prx.Param:
    """Create a fixed parameter."""
    value = _apply_wrappers(value, scale=scale, fixed=True)
    if not prx.is_param(value):
        value = prx.as_param(value)
    return value


def fixed(
    default: Any = dataclasses.MISSING,
    scale: float = 1.0,
) -> Any:
    """Create a parameter field specifier that is fixed by default."""
    if default is dataclasses.MISSING:
        default = 1.0
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else Fixed(x, scale)
    return eqx.field(default=default, converter=converter)

# ---------------------------------------------------------
# Positive
# ---------------------------------------------------------

def Positive(
    value: prx.Param,
    scale: float = 1.0,
    fixed: bool = False
) -> prx.Param:
    """Create a parameter constrained to be positive."""
    var = prx.Constrained(constraint=PositiveConstraint(), value=value)
    return _apply_wrappers(var, scale, fixed)

def positive(
    default: Any = dataclasses.MISSING,
    scale: float = 1.0,
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
    default: Any = dataclasses.MISSING,
    scale: float = 1.0,
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
    value: prx.Param | None = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
    if value is None:
        value = (lower + upper) / 2.0
    constrained_var = prx.Constrained(value=value, constraint=Interval(lower, upper))
    var = prx.Random(distribution=DistUniform(lower, upper), raw_value=constrained_var)
    return _apply_wrappers(var, scale, fixed)

def uniform(
    lower: float,
    upper: float,
    default: Any = dataclasses.MISSING,
    scale: float = 1.0,
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
    value: prx.Param | None = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> prx.Param:
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
    scale: float = 1.0,
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
    default: Any = dataclasses.MISSING,
    scale: float = 1.0,
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
    default: Any = dataclasses.MISSING,
    scale: float = 1.0,
    fixed: bool = False
) -> Any:
    if default is dataclasses.MISSING:
        default = mean
    def converter(x):
        return x if isinstance(x, prx.AbstractVariable) else RelativeNormal(mean, pct_std, x, scale, fixed)
    return eqx.field(default=default, converter=converter)