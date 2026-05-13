"""
Constraints for parameters.

Can be used for parameter factories in :mod:`pmrf.parameters`.

Builds on top of the library `Parax <https://gvcallen.github.io/parax>`_.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from parax.constraints import (
    AbstractConstraint as AbstractConstraint,
    RealLine as RealLine,
    Positive as Positive,
    Negative as Negative,
    Interval as Interval,
    GreaterThan as GreaterThan,
    LessThan as LessThan,
)


def intersect_constraints(a: AbstractConstraint, b: AbstractConstraint) -> AbstractConstraint:
    """
    Calculates the intersection of two constraints.
    Returns the most specific constraint class possible.
    """
    a_lower, a_upper = a.bounds
    b_lower, b_upper = b.bounds

    lower = jnp.maximum(a_lower, b_lower)
    upper = jnp.minimum(a_upper, b_upper)

    # Convert to concrete numpy arrays for boolean checks during init
    np_lower = jnp.asarray(lower)
    np_upper = jnp.asarray(upper)

    np_lower, np_upper = eqx.error_if(
        (np_lower, np_upper),
        jnp.any(jnp.greater_equal(np_lower, np_upper)),
        f"Constraint intersection is empty or invalid."
    )
    
    is_neginf_lower = jnp.all(jnp.isneginf(np_lower))
    is_posinf_upper = jnp.all(jnp.isposinf(np_upper))
    is_zero_lower = jnp.all(jnp.equal(np_lower, 0.0))
    is_zero_upper = jnp.all(jnp.equal(np_upper, 0.0))

    # Resolve to the most specific constraint class
    if is_neginf_lower and is_posinf_upper:
        return RealLine()
    elif is_zero_lower and is_posinf_upper:
        return Positive()
    elif is_neginf_lower and is_zero_upper:
        return Negative()
    elif is_posinf_upper:
        return GreaterThan(lower)
    elif is_neginf_lower:
        return LessThan(upper)
    else:
        return Interval(lower, upper)
    
    
__all__ = [
    'AbstractConstraint',
    'RealLine',
    'Positive',
    'Negative',
    'Interval',
    'GreaterThan',
    'LessThan',
    'intersect_constraints',
]