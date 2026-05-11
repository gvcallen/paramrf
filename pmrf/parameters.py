"""
Parameters and parameter field specifiers.

Builds on top of `parax`.
"""

import dataclasses
from typing import Any, Optional
from jaxtyping import ArrayLike

import jax.numpy as jnp
import equinox as eqx
import parax as prx
from parax.transforms import Scale
from distreqx.distributions import AbstractDistribution

from pmrf.constraints import AbstractConstraint, Interval, intersect_constraints
from pmrf.jax_utils import unwrap

Param = prx.Param

def _apply_wrappers(
    var: Param, 
    scale: ArrayLike = 1.0, 
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
    var = prx.as_param(var)

    if scale != 1.0:
        scale = jnp.asarray(scale, dtype=float)
        var = prx.Derived(Scale(scale), raw_value=var)
    if fixed:
        var = prx.Fixed(var)
    return var

# ---------------------------------------------------------
# Parameter Factories
# ---------------------------------------------------------

def Scaled(
    value: Param,
    scale: float,
    *,
    fixed: bool = False,
) -> Param:
    """
    Create a free parameter with optional scaling.
    """
    value = prx.as_param(value)
    return _apply_wrappers(value, scale=scale, fixed=fixed)


def Fixed(
    value: Param,
    *,
    scale: ArrayLike = 1.0,
) -> Param:
    """
    Create a fixed parameter that will not be optimized.
    """
    value = prx.as_param(value)
    return _apply_wrappers(value, scale=scale, fixed=True)


def Constrained(
    constraint: AbstractConstraint, 
    value: ArrayLike,
    *,
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter constrained within a specific interval.
    """
    value = jnp.asarray(value)
    var = prx.Constrained(constraint, value=value)
    return _apply_wrappers(var, scale, fixed)
    

def Bounded(
    lower: Any, 
    upper: Any, 
    *,
    value: Optional[ArrayLike] = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter constrained within a specific interval.
    """
    lower, upper = jnp.asarray(lower, dtype=float), jnp.asarray(upper, dtype=float)
    if value is None:
        value = (lower + upper) / 2.0
    return Constrained(Interval(lower, upper), value=value, scale=scale, fixed=fixed)


def Random(
    distribution: AbstractDistribution,
    *,
    constraint: Optional[AbstractConstraint] = None,
    value: Optional[ArrayLike] = None, 
    scale: float = 1.0, 
    fixed: bool = False
) -> Param:
    """
    Create a parameter constrained within a specific interval.
    """
    if value is None:
        try:
            value = distribution.mean()
        except:
            raise Exception("`value` was none when creating Random variable but `distribution` does not implement `mean`")
        
    value = jnp.asarray(value)
    var = prx.Random(distribution, constraint=constraint, value=value)
    return _apply_wrappers(var, scale, fixed)


# ---------------------------------------------------------
# Field Specifier
# ---------------------------------------------------------
    

def param(
    value: Any = dataclasses.MISSING,
    *,
    constraint: Optional[AbstractConstraint] = None,
    scale: float = 1.0,
    fixed: bool = False,
) -> Any:
    """
    A field specifier for defining the physical rules of model parameters.
    """
    def converter(x):
        # Respect fully formed variables and inject constraints if needed
        if prx.is_variable(x):
            if constraint is not None and prx.is_constrainable(x):
                combined_constraint = intersect_constraints(unwrap(constraint), unwrap(x.constraint))
                x = x.constrain(combined_constraint)
            return x

        # Build the default physical variable
        if constraint is not None:
            return Constrained(constraint=constraint, value=x, scale=scale, fixed=fixed)
        elif fixed:
            return Fixed(value=x, scale=scale)
        elif scale != 1.0:
            return Scaled(value=x, scale=scale)
        else:
            return jnp.asarray(x)
        
    return eqx.field(default=value, converter=converter)


__all__ = [
    "Scaled",
    "Fixed",
    "Positive",
    "Bounded",
    "Constrained",
    "Random",
    "param",
]