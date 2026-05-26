"""
Parameter factories, converters, and field specifiers.

Most of these are re-exported at root.

Builds on top of `Parax <https://gvcallen.github.io/parax>`_.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx
import parax as prx

from pmrf.constraints import AbstractConstraint, Interval
from pmrf.distributions import AbstractDistribution
from pmrf.types import Param
from pmrf.utils import unfreeze

def apply_wrappers(value: Any, scale: float = 1.0, fixed: bool = False, name: str | None = None) -> Param:
    value = prx.as_free(value)
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

    if name is not None:
        value = prx.Tagged(metadata={'name': name}, raw_value=value)

    return value


def extract_name(value: Any) -> str | None:
    """"
    Extracts a name from a value that might be a parameter.

    Parameters
    ----------
    value : Any, optional
        The value.

    Returns
    -------
    str | None
        The name if one was found, otherwise None.
    """
    if not prx.is_variable(value):
        return None
    
    def has_name(x):
        return isinstance(x, prx.Tagged) and 'name' in x.metadata
    
    named_variable = jax.tree.leaves(value, is_leaf=has_name)
    named_variable = [x for x in named_variable if has_name(x)]

    if len(named_variable) == 0:
        return None
    elif len(named_variable) > 1:
        raise Exception(f"Found multiple variables alongside with a name, which should be impossible. Value = {value}")
    
    return named_variable[0].metadata['name']
    

def as_free(
    value: Any = None,
    *,
    distribution: Optional[AbstractDistribution] = None,
    constraint: Optional[AbstractConstraint] = None,
    scale: float = 1.0,
    name: Optional[str] = None,
) -> Param:
    """
    Coerces a value into a free parameter.

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
    name : str, optional
        A name for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        A variable parameter.
    """
    # Unwrapping for safety
    distribution, constraint = prx.unwrap(distribution), prx.unwrap(constraint)
    
    # Error Checking & Value Inference
    if value is None and distribution is None and constraint is None:
        raise ValueError("`value` was None in `as_param` but neither a distribution nor a finite Interval constraint was provided")
    if distribution is not None and prx.is_variable(value):
        raise ValueError("Currently, you cannot assign a new distribution to an existing variable.")
    if constraint is not None and value is not None:
        value_array = jnp.array(value)
        eqx.error_if(
            value_array,
            constraint.is_outside(value_array),
            f"\n\nA parameter value falls outside the constraint ({value} is not in {constraint}). "
            f"\nMake sure the initial values match the parameter and model constraints.",
        )
        
    # Cater for none values
    if value is None:
        if constraint is not None and not jnp.any(jnp.isinf(jnp.asarray(constraint.bounds))):
            value = constraint.midpoint()
        else:
            value = distribution.mean()

    # Make sure the value is not fixed
    # TODO: this will not yet work for fixed parameters deeply composed within other variables
    value = unfreeze(value)
    
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

    return apply_wrappers(value, scale=scale, fixed=False, name=name)


def as_fixed(
    value: Any = None,
    *,
    distribution: Optional[AbstractDistribution] = None,
    constraint: Optional[AbstractConstraint] = None,
    scale: float = 1.0,
    name: Optional[str] = None,
) -> Param:
    """
    Coerces a value into a fixed parameter.

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
    name : str, optional
        A name for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        A fixed parameter.
    """
    variable = as_free(
        value,
        distribution=distribution,
        constraint=constraint,
        scale=scale,
        name=name,
    )

    return apply_wrappers(variable, scale=1.0, fixed=True, name=None)


def param(
    *,
    default: Any = dataclasses.MISSING,
    as_free: bool = False,
    as_fixed: bool = False,
    distribution: Optional[AbstractDistribution] = None,
    constraint: Optional[AbstractConstraint] = None,
    scale: float = 1.0,
    **kwargs,
) -> Any:
    """
    A field specifier for registering parameters within a model.

    This specifier can be used when declaring custom models inheriting from `pmrf.Model`.

    It is used to register the parameter within the model so that it is listed
    under :meth:`pmrf.Model.named_params`. It can also be used to enforce
    constraints, scaling, bounds and variability within the model itself.

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
    default : Any, optional
        The default value of the parameter.
    as_free : bool, optional
        Whether to enforce that the value is a variable parameter.
        If False, incoming values will keep the variability (e.g. constants will remain constants).
        If True, all values will be co-erced into variable parameters.
    as_fixed : bool, optional
        Whether to enforce that the value is a fixed parameter.
        If False, incoming values will keep the variability (e.g. constants will remain constants).
        If True, all values will be wrapped in :func:`pmrf.Fixed`.
    distribution : Optional[AbstractDistribution], optional
        The probability distribution for the parameter. See :mod:`pmrf.distributions`.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter. See :mod:`pmrf.constraints`.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    **kwargs
        Additional key-word arguments to pass to the general :func:`pmrf.field` specifier.

    Returns
    -------
    Any
        An equinox field with a built-in converter for parameter rules.
    """
    if as_fixed and as_free:
        raise ValueError("Cannot pass both `as_fixed=True` and `as_free=True` to `pmrf.param`")
    
    as_free_func = globals()['as_free']
    as_fixed_func = globals()['as_fixed']

    def converter(x):
        if as_free:
            return as_free_func(value=x, distribution=distribution, constraint=constraint, scale=scale)
        if as_fixed:
            return as_fixed_func(value=x, distribution=distribution, constraint=constraint, scale=scale)
        
        if x is None:
            return None
        elif (prx.is_variable(x) or isinstance(x, jax.Array)) and not prx.is_constant(x):
            return as_free_func(value=x, distribution=distribution, constraint=constraint, scale=scale)
        return as_fixed_func(value=x, distribution=distribution, constraint=constraint, scale=scale)

    return eqx.field(default=default, converter=converter, **kwargs)


def Fixed(
    value: ArrayLike,
    *,
    scale: float = 1.0,
    name: Optional[str] = None,
) -> Param:
    """
    Create a fixed parameter.

    Compared to specifying raw floats or numpy arrays, this is a convenience
    specifier that allows the parameters to be ignored by optimizers
    while still having a name and being capable of easily being made
    into a variable using :func:`pmrf.unfreeze`.

    Parameters
    ----------
    value : ArrayLike
        The parameter value to fix.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    name : str, optional
        A name for the parameter.       , by default None.

    Returns
    -------
    pmrf.Param
        The fixed parameter.
    """
    return as_fixed(value, scale=scale, name=name)


def Unconstrained(
    value: ArrayLike,
    *,
    scale: float = 1.0,
    fixed: bool = False,
    name: Optional[str] = None,
) -> Param:
    """
    Create an unconstrained free parameter.

    Parameters
    ----------
    value : ArrayLike
        The base parameter value.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
    fixed : bool, optional
        Wraps the parameter in a :class:`pmrf.Fixed` parameter.        
    name : str, optional
        A name for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        An unconstrained parameter.
    """
    p = as_free(value, scale=scale, name=name)
    return apply_wrappers(p, fixed=fixed)


def Constrained(
    constraint: AbstractConstraint, 
    value: ArrayLike,
    *,
    scale: float = 1.0, 
    fixed: bool = False,
    name: Optional[str] = None,
) -> Param:
    """
    Create a free parameter constrained to a specific domain.

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
        Wraps the parameter in a :class:`pmrf.Fixed` parameter.
    name : str, optional
        A name for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        The constrained parameter.
    """
    p = as_free(value, constraint=constraint, scale=scale, name=name)
    return apply_wrappers(p, fixed=fixed)


def Bounded(
    lower: Any, 
    upper: Any, 
    *,
    value: Optional[ArrayLike] = None, 
    scale: float = 1.0, 
    fixed: bool = False,
    name: Optional[str] = None,
) -> Param:
    """
    Create a free parameter constrained within a specific interval.

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
        Wraps the parameter in a :class:`pmrf.Fixed` parameter.
    name : str, optional
        A name for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        The bounded parameter.
    """
    p = as_free(value, constraint=Interval(lower, upper), scale=scale, name=name)
    return apply_wrappers(p, fixed=fixed)


def Random(
    distribution: AbstractDistribution,
    *,
    constraint: Optional[AbstractConstraint] = None,
    value: Optional[ArrayLike] = None, 
    scale: float = 1.0, 
    fixed: bool = False,
    name: Optional[str] = None,
) -> Param:
    """
    Create a free parameter with an associated probability distribution.

    Used as the main factory to define parameters for Bayesian inference.
    Can also be used for bounded optimization, in which case the random
    variable's domain (constraint) is used as the bounds.

    For built-in distributions, see :mod:`pmrf.distributions`.
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
        Wraps the parameter in a :class:`pmrf.Fixed` parameter.
    name : str, optional
        A name for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        The random parameter.

    Raises
    ------
    ValueError
        If `value` is None and the distribution does not implement `mean()`.
    """
    p = as_free(value, distribution=distribution, constraint=constraint, scale=scale, name=name)
    return apply_wrappers(p, fixed=fixed)

__all__ = [
    "as_free",
    "as_fixed",
    "Unconstrained",
    "Fixed",
    "Bounded",
    "Constrained",
    "Random",
    "Param",
    "param",
    "extract_name",
]