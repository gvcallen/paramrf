"""
Parameter factory functions and the main field specifier.

Builds on top of the library `Parax <https://gvcallen.github.io/parax>`_.
"""
from functools import partial
import warnings
import dataclasses
from typing import Any, Optional
from jaxtyping import ArrayLike

import jax.numpy as jnp
import equinox as eqx
import parax as prx
from parax.transforms import Scale

from pmrf.constraints import AbstractConstraint, Interval, intersect_constraints
from pmrf.distributions import AbstractDistribution, truncate_distribution

Param = prx.Param
"""The abstract Parameter type hint for parameters in models."""

# ---------------------------------------------------------
# The Core Engine (Exposed in API)
# ---------------------------------------------------------

def apply_wrappers(value: Any, scale: float, fixed: bool):
    value = prx.as_variable(value)
    if scale != 1.0:
        scale_val = jnp.asarray(scale, dtype=float)
        value = prx.Derived(Scale(scale_val), raw_value=value)
    if fixed:
        value = prx.Fixed(value)    
    return value
    

def as_param(
    value: Any = None,
    *,
    distribution: Optional[AbstractDistribution] = None,
    constraint: Optional[AbstractConstraint] = None,
    scale: float = 1.0,
    fixed: bool = False
) -> Param:
    """
    Coerces a value into a parameter.

    Parameters
    ----------
    value : Any, optional
        The value of the parameter.
    distribution : Optional[AbstractDistribution], optional
        The probability distribution for the parameter. See :mod:`pmrf.distributions`.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter. See :mod:`pmrf.constraints`.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.

    Returns
    -------
    Any
        An equinox field with a built-in converter for parameter rules.
    """
    # Check invalid cases
    if distribution is not None and prx.is_variable(value):
        raise ValueError("Currently, you cannot assign a new distribution to an existing variable.")
    
    if value is None:
        if distribution is not None:
            try:
                value = distribution.mean()
            except Exception as e:
                raise ValueError("`value` was None but the provided `distribution` does not implement `mean()`") from e
        else:
            raise ValueError("`value` was none in `as_param`")
    
    if prx.is_variable(value) and constraint is not None and not prx.is_constrainable(value):
        # If a variable is provided that is not constrainable but IS wrappable,
        # we can push a constrained variable INSIDE that value
        if prx.is_wrappable(value):
            value = value.wrap(prx.Constrained(constraint, value=jnp.array(value)))
            return apply_wrappers(value, scale=scale, fixed=fixed)
        else:
            raise ValueError(f"A constraint was specified, but the existing variable is not constrainable no wrappable. Value = {value}")
    
    # Unwrap inputs and default-construct random/constrained variables
    distribution, constraint = prx.unwrap(distribution), prx.unwrap(constraint)
    if not prx.is_variable(value):
        if distribution is not None:
            value = prx.Random(distribution, value=value)
        elif constraint is not None:
            value = prx.Constrained(constraint, value=value)
        else:
            value = prx.Real(value)
    
    # Combine constraints if our input is constrainable and we were passed a new constraint
    if prx.is_constrainable(value) and constraint is not None:
        orig_constraint = prx.unwrap(value.constraint)
        constraint = intersect_constraints(constraint, orig_constraint)
        orig_lower, orig_upper = orig_constraint.bounds
        has_shrunk = jnp.any(constraint.bounds[0] > orig_lower) or jnp.any(constraint.bounds[1] < orig_upper)
        
        # Truncate an existing random variable's distribution, if necessary
        if isinstance(value, prx.Random) and has_shrunk:
            try:
                new_lower, new_upper = constraint.bounds
                trunc_dist = truncate_distribution(prx.unwrap(value.distribution), new_lower, new_upper)
                trunc_value = jnp.clip(jnp.array(value), min=new_lower, max=new_upper)
                value = prx.Random(trunc_dist, constraint=constraint, value=trunc_value)
            except Exception:
                value = value.constrain(constraint)
                dist_name = type(prx.unwrap(value.distribution)).__name__
                warnings.warn(
                    f"A constraint was applied, but the prior distribution ({dist_name}) "
                    f"could not be automatically truncated and will therefore be warped. "
                    f"It is recommended to choose a distribution that aligns with the physical constraints.",
                    UserWarning, stacklevel=2
                )
        else:
            value = value.constrain(constraint)
                
    value = apply_wrappers(value, scale=scale, fixed=fixed)
    return value


# ---------------------------------------------------------
# Field Specifier
# ---------------------------------------------------------

def param(
    value: Any = dataclasses.MISSING,
    *,
    distribution: Optional[AbstractDistribution] = None,
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
    distribution : Optional[AbstractDistribution], optional
        The probability distribution for the parameter. See :mod:`pmrf.distributions`.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter. See :mod:`pmrf.constraints`.
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
        return as_param(value=x, distribution=distribution, constraint=constraint, scale=scale, fixed=fixed)
    
    return eqx.field(default=value, converter=converter)


# ---------------------------------------------------------
# Parameter Factories (Syntactic Sugar)
# ---------------------------------------------------------

def Free(
    value: ArrayLike,
) -> Param:
    """
    Create a simple free parameter.

    Parameters
    ----------
    value : ArrayLike
        The base parameter value.

    Returns
    -------
    Param
        The unconstrained parameter.
    """
    return as_param(value)


def Scaled(
    value: ArrayLike,
    scale: float,
    *,
    fixed: bool = False,
) -> Param:
    """
    Create a free parameter with scaling.

    Parameters
    ----------
    value : ArrayLike
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
    return as_param(value, scale=scale, fixed=fixed)


def Fixed(
    value: ArrayLike,
    *,
    scale: float = 1.0,
) -> Param:
    """
    Create a fixed parameter that will not be optimized.

    Parameters
    ----------
    value : ArrayLike
        The parameter value to fix.
    scale : float, optional
        The scaling factor to apply, by default 1.0.

    Returns
    -------
    Param
        The fixed parameter.
    """
    return as_param(value, scale=scale, fixed=True)


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
    return as_param(value, constraint=constraint, scale=scale, fixed=fixed)


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
    return as_param(value, constraint=Interval(lower, upper), scale=scale, fixed=fixed)


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
    ValueError
        If `value` is None and the distribution does not implement `mean()`.
    """
    return as_param(value, distribution=distribution, constraint=constraint, scale=scale, fixed=fixed)


__all__ = [
    "as_param",
    "Free",
    "Scaled",
    "Fixed",
    "Bounded",
    "Constrained",
    "Random",
    "param",
]