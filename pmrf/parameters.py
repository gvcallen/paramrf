"""
Parameter factories and parameter field specifiers.

Builds on top of the library `Parax <https://gvcallen.github.io/parax>`_.
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
"""The main type hint for parameters in models."""


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

    Parameters
    ----------
    value : Param
        The base parameter value.
    scale : float
        The scaling factor to apply.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.

    Returns
    -------
    Param
        The parameter wrapped with scaling (and optionally fixed).
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

    Parameters
    ----------
    value : Param
        The parameter value to fix.
    scale : ArrayLike, optional
        The scaling factor to apply, by default 1.0.

    Returns
    -------
    Param
        The fixed parameter.
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
    Create a parameter constrained to a specific domain.

    Parameters
    ----------
    constraint : AbstractConstraint
        The constraint to apply to the parameter.
    value : ArrayLike
        The initial value of the parameter.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.

    Returns
    -------
    Param
        The constrained parameter.
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

    Parameters
    ----------
    lower : Any
        The lower bound of the interval.
    upper : Any
        The upper bound of the interval.
    value : Optional[ArrayLike], optional
        The initial value. If None, the midpoint of the bounds is used.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.

    Returns
    -------
    Param
        The bounded parameter.
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
    Create a parameter initialized from a random distribution.

    Parameters
    ----------
    distribution : AbstractDistribution
        The probability distribution for the parameter.
    constraint : Optional[AbstractConstraint], optional
        An optional constraint to apply.
    value : Optional[ArrayLike], optional
        The initial value. If None, the distribution's mean is used.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.

    Returns
    -------
    Param
        The random parameter.

    Raises
    ------
    Exception
        If `value` is None and the distribution does not implement `mean`.
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

    Parameters
    ----------
    value : Any, optional
        The default value of the field.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.

    Returns
    -------
    Any
        An equinox field with a built-in converter for parameter rules.
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