"""
Parameter factories, converters, and field specifiers.

Note: Many of these utilities are re-exported at root.

Builds on top of `Parax <https://gvcallen.github.io/parax>`_.
"""
import dataclasses
from typing import Any, Optional, TypeAlias
from jaxtyping import ArrayLike, Inexact, Array

import jax.numpy as jnp
import equinox as eqx
import parax as prx

from pmrf.constraints import AbstractConstraint, Interval
from pmrf.distributions import AbstractDistribution

#: The abstract Parameter type hint for parameters in models.
Param: TypeAlias = prx.AbstractVariable | Inexact[Array, "..."]

# ---------------------------------------------------------
# The Core Engine (Exposed in API)
# ---------------------------------------------------------

def apply_wrappers(value: Any, scale: float, fixed: bool):
    value = prx.as_variable(value)
    if scale != 1.0:
        scale_val = jnp.asarray(scale, dtype=float)
        try:
            from distreqx.bijectors import Scale
            bij = Scale(scale_val)
        except:
            from distreqx.bijectors import ScalarAffine
            bij = ScalarAffine(shift=jnp.zeros_like(scale_val), scale=scale_val)
            
        value = prx.Transformed(bij, raw_value=value)
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

    The incoming value can be an existing parameter or any
    parameter-like object (float, array etc.).

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
    # Unwrapping for safety
    distribution, constraint = prx.unwrap(distribution), prx.unwrap(constraint)
    
    # Error Checking & Value Inference
    if value is None and distribution is None and constraint is None:
        raise ValueError("`value` was None in `as_param` but neither a distribution nor a finite Interval constraint was providied")
    if distribution is not None and prx.is_variable(value):
        raise ValueError("Currently, you cannot assign a new distribution to an existing variable.")
    if constraint is not None and value is not None and constraint.is_outside(jnp.array(value)):
        raise ValueError(
            f"\n\nA parameter value falls outside the constraint ({value} is not in {constraint}). "
            f"\nMake sure the initial values match the parameter and model constraints."
        )
        
    # Cater for none values
    if value is None:
        if constraint is not None and not jnp.any(jnp.isinf(jnp.asarray(constraint.bounds))):
            value = constraint.midpoint()
        else:
            value = distribution.mean()
    
    # Base Construction (if it's not a variable yet)
    if prx.is_variable(value):
        constraints = [constraint] if constraint is not None else []
        if prx.is_constrained(value):
            constraints.append(prx.unwrap(value.constraint))
        if len(constraints) != 0:
            value = prx.variables.constrain_param(value, *constraints)
    else:
        if distribution is not None:
            value = prx.Random(distribution, constraint=constraint, value=value)
        elif constraint is not None:
            value = prx.Constrained(constraint, value=value)
        else:
            value = prx.Real(value)

    return apply_wrappers(value, scale=scale, fixed=fixed)

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
    A field specifier for defining the rules of parameters in custom model.

    This specifier can be used when declaring custom models inheriting from `pmrf.Model`.
    For example, it can be used to enforce constraints/scaling/bounds that are required
    by the model itself.

    Example
    --------

    Declaring a parameter with a positive constraint and built-in scale:

    .. code-block:: python

        import pmrf as prf
        from pmrf.models import Resistor, Capacitor
        from pmrf.constraints import Positive

        class RC(prf.Model):
            R: prf.Param = prf.param(constraint=Positive())
            C: prf.Param = prf.param(constraint=Positive(), scale=1e-12)

            def build(self) -> prf.Model:
                return Resistor(self.R) ** Capacitor(self.C)

        RC(1.0, 2.0)
        # RC(R=1., C=2.e-12)

        RC(-1.0, 2.0)
        # ValueError: out of bounds

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

def Value(
    value: ArrayLike,
    *,
    scale: float = 1.0,
    fixed: bool = False,
) -> Param:
    """
    Create a simple parameter with an optional scale.

    Parameters
    ----------
    value : ArrayLike
        The base parameter value.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Whether to freeze the parameter, by default False.        

    Returns
    -------
    Param
        An unconstrained parameter.
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

    See :mod:`pmrf.constraints` for built-in constraints.

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

    Used as the main factory to define parameters for bounded optimization.

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
    Create a parameter initialized with a random distribution.

    Used as the main factory to define parameters for Bayesian inference.
    Can also be used for bounded optimization, in which case the random
    variable's domain (constraint) is used as the bounds.

    For built-in distributions, see :mod:`pmrf.distribution`.
    For built-in constraints, see :mod:`pmrf.constraints`.

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
    "Value",
    "Fixed",
    "Bounded",
    "Constrained",
    "Random",
    "param",
]