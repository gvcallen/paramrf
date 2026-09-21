"""
Parameter factories, converters, and field specifiers.

Most of these are re-exported at root.

Builds on top of `Parax <https://gvcallen.github.io/parax>`_.
"""
from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Mapping
from typing import Any, Literal, Optional, Self, Sequence, Union, Callable, TypeVar, TypeGuard

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Array
import equinox as eqx
import parax as prx
from parax.annotation import AbstractAnnotated

from pmrf.bijectors import AbstractBijector, Chain, ScalarAffine
from pmrf.constraints import AbstractConstraint, Interval
from pmrf.distributions import AbstractDistribution, Transformed
from pmrf.utils import error_if, field
from pmrf.utils.tree import Pathgetter, path_nodes, path_to_name, resolve_target


T = TypeVar('T')


class Param(prx.AbstractVariable, AbstractAnnotated[Any]):
    """
    The canonical parameter container for ParamRF.

    Parameters can be created by instantiating this class, or using factories
    in :mod:`pmrf.parameters`, most of which are re-exported at root
    (e.g. :func:`pmrf.Unconstrained`, :func:`pmrf.Fixed`, :func:`pmrf.Bounded`).
    
    Wraps a `Parax <https://gvcallen.github.io/parax>`_ variable,
    applying an optional scale, name and metadata.

    A parameter's number lives in one of three spaces:

    - **declared**: the number as written, in the units the scale declares
      (:attr:`value`). Construction, bounds and distributions are in declared space.
    - **physical**: the declared value times the scale (:attr:`physical_value`).
      Unwrapping and arithmetic use the physical value.
    - **raw**: the latent array an optimiser or sampler moves through
      (:attr:`raw_value`).

    Examples
    --------
    .. code-block:: python

        p = prf.Unconstrained(2.0, scale=1e-12)
        p.value            # 2.0
        p.physical_value   # 2e-12
    """
    #: The Parax variable holding the declared value, constraint and distribution.
    variable: prx.AbstractVariable = eqx.field(converter=prx.as_variable)

    #: The units the declared value is written in: physical = declared × scale.
    #: None means not set, which acts as 1.0 and lets a field's scale apply.
    scale: float | None = eqx.field(converter=lambda s: None if s is None else float(s), default=None, static=True)
    
    #: A name for the parameter.
    name: str | None = field(default=None, kw_only=True, static=True)

    #: Arbitrary metadata to store alongside the parameter.
    metadata: Any = field(default=None, kw_only=True, static=True)

    def __init__(
        self,
        *,
        value: Optional[ArrayLike] = None,
        distribution: Optional[AbstractDistribution] = None,
        constraint: Optional[AbstractConstraint] = None,
        name: Optional[str] = None,
        scale: Optional[float] = None,
        fixed: bool = False,
        metadata: Any = None,
        variable: Optional[prx.AbstractVariable] = None,
    ):
        """
        Creates a generic parameter.

        The incoming value can be any ArrayLike object.

        Parameters
        ----------
        value : ArrayLike, optional
            The declared value of the parameter.
        distribution : Optional[AbstractDistribution], optional
            The probability distribution, in declared space. See :mod:`pmrf.distributions`.
        constraint : Optional[AbstractConstraint], optional
            The constraint, in declared space. See :mod:`pmrf.constraints`.
        name : str, optional
            A name for the parameter, by default None.
        scale : float, optional
            The units the value is written in. None, the default, acts as 1.0 and
            lets a field declared with :func:`pmrf.param` apply its own scale.
        fixed : bool, optional
            Initializes the parameter as fixed. Defaults to False.
        metadata : Any, optional
            Arbitrary metadata for the parameter, by default None.
        variable : Optional[prx.AbstractVariable], optional
            The Parax variable to wrap. If `value` is also passed, the variable's
            declared value is replaced, keeping its distribution, constraint, fixed
            state, and the dtype, shape and `weak_type` of its array, so the jit
            cache key is unchanged. This is what makes
            ``pmrf.replace(param, value=...)`` work, so
            ``pmrf.replace(param, value=param.value)`` is the identity (up to the
            floating-point round trip through the constraint bijector).
        """
        if isinstance(value, prx.AbstractVariable):
            raise ValueError("Got a Parax variable when constructing a parameter")

        if variable is not None and (distribution is not None or constraint is not None):
            raise ValueError("Cannot pass `variable` with `distribution` or `constraint` to Param constructor")

        if variable is not None and value is not None:
            variable = _replace_variable_value(variable, value)
        elif variable is None:
            distribution, constraint = prx.unwrap(distribution), prx.unwrap(constraint)
            
            # Error Checking & Value Inference
            if value is None and distribution is None and constraint is None:
                raise ValueError("`value` was None when constructing a parameter but neither a distribution nor a finite Interval constraint was provided")
            if constraint is not None and value is not None:
                value = _check_in_constraint(value, constraint)

            # Cater for none values
            if value is None:
                if constraint is not None and not jnp.any(jnp.isinf(jnp.asarray(constraint.bounds))):
                    value = constraint.midpoint()
                else:
                    value = distribution.mean()

            value = jnp.asarray(value)
            if distribution is not None:
                variable = prx.Random(distribution, constraint=constraint, value=value)
            elif constraint is not None:
                variable = prx.Constrained(constraint, value=value)
            else:
                variable = prx.Real(value)
            raw = variable.raw_value
            if (isinstance(raw, jax.Array) and jnp.issubdtype(value.dtype, jnp.floating)
                    and raw.dtype != value.dtype):
                # Some Parax bijectors clip with a float32 epsilon, which demotes a
                # weak float64 value. Raw values keep the declared dtype, so a solver
                # moving several of them sees one dtype.
                variable = eqx.tree_at(lambda v: v.raw_value, variable, raw.astype(value.dtype))
            if fixed:
                variable = prx.Fixed(variable)

        self.variable = variable
        self.scale = None if scale is None else float(scale)
        self.name = name
        self.metadata = metadata

    @property
    def fixed(self) -> bool:
        """
        Indicates whether the parameter is fixed (constant).

        Returns
        -------
        bool
            True if the parameter is fixed, False otherwise.
        """
        return prx.is_constant(self.variable)

    @property
    def as_fixed(self):
        # Parax's `AbstractVariable.as_fixed` would wrap the parameter in a Parax
        # `Fixed`, losing its name and scale. Fixed state is set with `pmrf.update`.
        raise AttributeError("'Param' has no attribute 'as_fixed'; use `pmrf.update(param, fixed=True)`.")

    @property
    def distribution(self) -> AbstractDistribution | None:
        """
        The probability distribution of the parameter, in declared space.

        Returns
        -------
        AbstractDistribution | None
            The distribution if one exists, otherwise None.
        """
        if prx.is_probabilistic(self.variable):
            return self.variable.distribution
        return None

    @property
    def constraint(self) -> AbstractConstraint | None:
        """
        The constraint of the parameter, in declared space.

        Returns
        -------
        AbstractConstraint | None
            The constraint if one exists, otherwise None.
        """
        if prx.is_constrained(self.variable):
            return self.variable.constraint
        return None

    @property
    def bounds(self) -> tuple[ArrayLike, ArrayLike] | None:
        """
        The lower and upper bounds of the parameter, in declared space.

        Returns
        -------
        tuple[ArrayLike, ArrayLike] | None
            A tuple of (lower_bound, upper_bound) if bounds exist, otherwise None.
        """
        if prx.is_bounded(self.variable):
            return self.variable.bounds
        return None

    @property
    def raw_to_declared_bijector(self) -> AbstractBijector | None:
        """
        The bijector mapping the raw value to the declared value.

        Raw space is what Parax calls unconstrained space. ParamRF calls it raw to
        avoid confusion with :func:`pmrf.Unconstrained`, which creates a parameter
        without bounds.

        Returns
        -------
        AbstractBijector | None
            The bijector if a constraint exists, otherwise None.
        """
        if self.constraint is None:
            return None
        return prx.as_unwrapped(self.constraint).bijector

    @property
    def declared_to_physical_bijector(self) -> AbstractBijector | None:
        """
        The bijector mapping the declared value to the physical value.

        This is multiplication by the scale. Distributions and bounds are in declared
        space, so this is the step needed to compare them against an unwrapped,
        physical value.

        Returns
        -------
        AbstractBijector | None
            The bijector if the parameter is scaled, otherwise None.
        """
        if self._scale == 1.0:
            return None
        return ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(self._scale))

    @property
    def bijector(self) -> AbstractBijector | None:
        """
        The full bijector mapping the raw value to the physical value.

        Composes :attr:`raw_to_declared_bijector` with
        :attr:`declared_to_physical_bijector`.

        Returns
        -------
        AbstractBijector | None
            The bijector if a constraint exists, otherwise None.
        """
        raw_to_declared = self.raw_to_declared_bijector
        if raw_to_declared is None:
            return None
        declared_to_physical = self.declared_to_physical_bijector
        if declared_to_physical is None:
            return raw_to_declared
        return Chain([declared_to_physical, raw_to_declared])

    @property
    def _scale(self) -> float:
        return 1.0 if self.scale is None else self.scale

    @property
    def value(self) -> jax.Array:
        """
        The value in declared space: the number as written, in the parameter's units.

        Returns
        -------
        jax.Array
            The declared value.
        """
        return jnp.asarray(self.variable)

    @property
    def physical_value(self) -> jax.Array:
        """
        The value in physical space: the declared value times the scale.

        This is what :func:`pmrf.unwrap` and arithmetic on the parameter use.

        Returns
        -------
        jax.Array
            The physical value.
        """
        if self._scale == 1.0:
            return self.value
        return self.value * self._scale

    @property
    def raw_value(self) -> jax.Array:
        """
        The value in raw space: the latent array an optimiser or sampler moves through.

        Returns
        -------
        jax.Array
            The raw array.
        """
        # Parax names a variable's latent array `raw_value`; `Fixed` holds a variable there.
        return jnp.asarray(_peel_fixed(self.variable).raw_value)

    def unwrap(self) -> jax.Array:
        """
        Unwraps the parameter to its value in physical space.

        Returns
        -------
        jax.Array
            The physical value, as :attr:`physical_value`.
        """
        return self.physical_value

    def __jax_array__(self) -> jax.Array:
        """Converts the parameter to an array of its physical value."""
        return self.physical_value

    @property
    def raw_leaf(self) -> jax.Array | None:
        """
        Returns this parameter's single remaining raw/whitened array leaf,
        without relying on the internal structure of the wrapped Parax
        variable (e.g. `Random`, `Fixed`). Intended for use after masking or
        partitioning has already reduced this parameter's own metadata (its
        distribution, constraint, etc.) to None, leaving only its own raw
        value behind.

        Returns
        -------
        jax.Array | None
            The one remaining leaf, or None if none remain (e.g. this
            parameter was masked out entirely).

        Raises
        ------
        ValueError
            If more than one leaf remains, e.g. called before masking.
        """
        leaves = jax.tree_util.tree_leaves(self)
        if len(leaves) == 0:
            return None
        if len(leaves) != 1:
            raise ValueError(f"Expected at most one remaining leaf, got {len(leaves)}: {leaves}")
        return leaves[0]

    def _wrap(self, value: Array) -> Self:
        """
        Updates the internal state of the parameter using a physical value.

        Parameters
        ----------
        value : Array
            The physical value to wrap.

        Returns
        -------
        Self
            A new instance of the parameter with the updated state.
            
        Raises
        ------
        ValueError
            If the underlying Parax variable is not wrappable.
        """
        if not prx.is_wrappable(self.variable):
            raise ValueError("Cannot wrap a parameter that wraps a non-wrappable Parax variable")

        new_variable = self.variable.wrap(value / self._scale)
        return eqx.tree_at(lambda x: x.variable, self, new_variable)


def _physical_operator(name: str):
    def operator(self, *args):
        return getattr(self.physical_value, name)(*args)
    operator.__name__ = name
    return operator


# Parax's arithmetic operators read `value`, which is declared for a `Param`. A
# parameter in an expression stands for its physical quantity, so they are rebound.
for _name in (
    "__getitem__", "__len__", "__iter__", "__contains__",
    "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__", "__floordiv__",
    "__mod__", "__divmod__", "__pow__", "__radd__", "__rsub__", "__rmul__",
    "__rmatmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rdivmod__",
    "__rpow__", "__neg__", "__pos__", "__abs__", "__invert__", "__complex__",
    "__int__", "__float__", "__index__", "__round__",
):
    setattr(Param, _name, _physical_operator(_name))
del _name


def _check_in_constraint(value: ArrayLike, constraint: AbstractConstraint) -> Array:
    """Returns `value` as an array that raises at runtime if it is outside `constraint`.

    The returned array must be used: under `jax.jit` the check lives only in its graph.
    """
    value_array = jnp.asarray(value)
    return error_if(
        value_array,
        constraint.is_outside(value_array),
        f"\n\nA parameter value falls outside the constraint ({value} is not in {constraint}). "
        f"\nMake sure the values match the parameter and model constraints.",
    )


def _peel_fixed(variable: prx.AbstractVariable) -> prx.AbstractVariable:
    """Returns the variable a `parax.Fixed` holds, or `variable` if it is not fixed."""
    return variable.raw_value if isinstance(variable, prx.Fixed) else variable


def _like(new: Any, old: Any) -> Any:
    """Casts array `new` to the dtype and `weak_type` of array `old`, which with shape
    is what `eqx.filter_jit` keys on. Non-array leaves are returned as they are."""
    if not isinstance(old, jax.Array) or not isinstance(new, jax.Array):
        return new
    # JAX has no public way to set `weak_type`.
    from jax._src.lax.lax import _convert_element_type
    return _convert_element_type(new, old.dtype, old.weak_type)


def _replace_variable_value(variable: prx.AbstractVariable, value: ArrayLike) -> prx.AbstractVariable:
    """Replaces the declared value of a Parax variable, keeping its structure.

    `Fixed` is peeled off before wrapping, since wrapping through it drops the wrapped
    variable's distribution, and the value is checked against the constraint because
    wrapping does not check bounds. Every array leaf keeps its dtype and `weak_type`,
    so the jit cache key is unchanged.
    """
    fixed = isinstance(variable, prx.Fixed)
    inner = _peel_fixed(variable)
    value_array = jnp.asarray(value)
    if prx.is_constrained(inner):
        value_array = _check_in_constraint(value_array, prx.unwrap(inner.constraint))
    new_inner = jax.tree.map(_like, inner.wrap(value_array), inner)
    return prx.Fixed(new_inner) if fixed else new_inner


def is_param(x: Any) -> TypeGuard[Param]:
    """
    Returns if `x` is an instance of :class:`pmrf.Param`.
    """
    return isinstance(x, Param)


def as_param(
    value: Any = None,
    *,
    constraint: Optional[AbstractConstraint] = None,
    scale: Optional[float] = None,
    as_free: bool = False,
    as_fixed: bool = False,
) -> Param:
    """
    Coerces a value into a parameter.

    The incoming value can be an existing parameter or parax variable,
    or any parameter-like object (float, array etc.).

    Constraints are intersected. A parameter's own scale overrides `scale`; the
    two are never multiplied.

    Parameters
    ----------
    value : Any, optional
        The declared value of the parameter.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter, in declared space. See :mod:`pmrf.constraints`.
    scale : float, optional
        The units `value` is written in, used unless `value` is a parameter with its
        own scale. None, the default, leaves the scale unset (acting as 1.0).
    as_free : bool, optional
        Whether to enforce that the value is a free parameter.
        If False, incoming values will keep the variability (e.g. constants will remain constants).
        If True, all values will be co-erced into free parameters.
    as_fixed : bool, optional
        Whether to enforce that the value is a fixed parameter.
        If False, incoming values will keep the variability (e.g. constants will remain constants).
        If True, all values will be wrapped in :func:`pmrf.Fixed`.

    Returns
    -------
    pmrf.Param
        A fixed parameter.
    """
    if as_free and as_fixed:
        raise ValueError("Cannot pass both `as_free=True` and `as_fixed=True`.")

    # Intersect parameter properties
    name = None
    metadata = None
    if isinstance(value, Param):
        if value.scale is not None:
            scale = value.scale
        name = value.name
        metadata = value.metadata
        value = value.variable

    # Intersect variable properties
    distribution = None
    fixed = None
    if prx.is_variable(value):
        if isinstance(value, prx.Fixed):
            fixed = True
            value = value.raw_value
        else:
            fixed = False
        
        if isinstance(value, prx.Random):
            distribution = value.distribution
        
        if prx.is_constrained(value):
            constraints = [constraint] if constraint is not None else []
            if prx.is_constrained(value):
                constraints.append(prx.unwrap(value.constraint))
            if len(constraints) != 0:
                value = prx.variables.constrain_param(value, *constraints)
            constraint = value.constraint
            
        if not isinstance(value, prx.Random | prx.Constrained | prx.Real):
            raise ValueError(f"Got unknown type in `as_param`: {value}")
            
        value = jnp.asarray(value)

    # Intersect fixed properties
    if fixed is None:
        if isinstance(value, jnp.ndarray):
            fixed = False
        else:
            fixed = True

    # Create the new parameter
    p = Param(
        value=value,
        distribution=distribution,
        constraint=constraint,
        scale=scale,
        fixed=fixed,
        name=name,
        metadata=metadata,
    )

    if as_fixed:
        p = _with_fixed(p, True)
    if as_free:
        p = _with_fixed(p, False)
    return p
    

def param(
    *,
    default: Any = dataclasses.MISSING,
    as_free: bool = False,
    as_fixed: bool = False,
    constraint: Optional[AbstractConstraint] = None,
    scale: Optional[float] = None,
    **kwargs,
) -> Any:
    """
    A field specifier for registering parameters within a model.

    This specifier can be used when declaring custom models inheriting from `pmrf.Model`.

    It is used to register the parameter when a model is constructed, so it is listed
    by :func:`pmrf.params`. It can also be used to enforce
    constraints, scaling, bounds and variability within the model itself.
    
    This simply creates a `pmrf.field` with a `pmrf.as_param` converter.
    
    Example
    --------

    Declaring a parameter with a positive constraint and built-in scale:

    .. code-block:: python

        import pmrf as prf
        from pmrf.constraints import Positive

        class RC(prf.Module):
            R: prf.Param = prf.param(constraint=Positive())
            C: prf.Param = prf.param(constraint=Positive(), scale=1e-12)

        rc = RC(1.0, 2.0)
        rc.C.value            # 2.0, in pF
        rc.C.physical_value   # 2e-12

        RC(1.0, prf.Unconstrained(2.0, scale=1e-9)).C.physical_value
        # 2e-09: an explicit scale overrides the field's

        RC(-1.0, 2.0)
        # ValueError: out of bounds

    Parameters
    ----------
    default : Any, optional
        The default value of the parameter.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter, in declared space. See :mod:`pmrf.constraints`.
    scale : float, optional
        The units values of this field are written in, by default None (1.0). A
        parameter passed with its own scale keeps it.
    as_free : bool, optional
        Whether to enforce that the value is a variable parameter.
        If False, incoming values will keep the variability (e.g. constants will remain constants).
        If True, all values will be co-erced into variable parameters.
    as_fixed : bool, optional
        Whether to enforce that the value is a fixed parameter.
        If False, incoming values will keep the variability (e.g. constants will remain constants).
        If True, all values will be wrapped in :func:`pmrf.Fixed`.
    **kwargs
        Additional key-word arguments to pass to the general :func:`pmrf.field` specifier.

    Returns
    -------
    Any
        An equinox field with a built-in converter for parameter rules.
    """
    def converter(x):
        return as_param(
            value=x,
            constraint=constraint,
            scale=scale,
            as_free=as_free,
            as_fixed=as_fixed,
        )

    return eqx.field(default=default, converter=converter, **kwargs)


def Fixed(
    value: ArrayLike,
    *,
    name: Optional[str] = None,
    scale: Optional[float] = None,
    metadata: Optional[Any] = None,
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
        The initial declared value.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The units the value is written in. None, the default, inherits the scale of
        the field the parameter is passed to, or acts as 1.0.
    metadata : Any, optional
        Arbitrary metadata for the parameter, by default None.        


    Returns
    -------
    pmrf.Param
        The fixed parameter.
    """
    return Param(value=value, scale=scale, name=name, fixed=True, metadata=metadata)


def Unconstrained(
    value: ArrayLike,
    *,
    fixed: bool = False,
    scale: Optional[float] = None,
    name: Optional[str] = None,
    metadata: Optional[Any] = None,
) -> Param:
    """
    Create an unconstrained free parameter.

    Parameters
    ----------
    value : ArrayLike
        The initial declared value.
    fixed : bool, optional
        Wraps the parameter in a :class:`pmrf.Fixed` parameter.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The units the value is written in. None, the default, inherits the scale of
        the field the parameter is passed to, or acts as 1.0.
    metadata : Any, optional
        Arbitrary metadata for the parameter, by default None.        

    Returns
    -------
    pmrf.Param
        An unconstrained parameter.
    """
    return Param(value=value, scale=scale, name=name, fixed=fixed, metadata=metadata)


def Constrained(
    constraint: AbstractConstraint, 
    value: ArrayLike,
    *,
    fixed: bool = False,
    name: Optional[str] = None,
    scale: Optional[float] = None,
    metadata: Optional[Any] = None,
) -> Param:
    """
    Create a free parameter constrained to a specific domain.

    See :mod:`pmrf.constraints` for built-in constraints.

    Parameters
    ----------
    constraint : AbstractConstraint
        The constraint to apply to the parameter.
    value : ArrayLike
        The initial declared value.
    fixed : bool, optional
        Initializes the parameter as fixed. Defaults to False.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The units the value is written in. None, the default, inherits the scale of
        the field the parameter is passed to, or acts as 1.0.
    metadata : Any, optional
        Arbitrary metadata for the parameter, by default None.        


    Returns
    -------
    pmrf.Param
        The constrained parameter.
    """
    return Param(value=value, constraint=constraint, scale=scale, name=name, fixed=fixed, metadata=metadata)


def Bounded(
    lower: Any, 
    upper: Any, 
    *,
    value: Optional[ArrayLike] = None, 
    fixed: bool = False,
    name: Optional[str] = None,
    scale: Optional[float] = None,
    metadata: Optional[Any] = None,
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
        The initial declared value. If None, the midpoint of the bounds is used.
    fixed : bool, optional
        Initializes the parameter as fixed. Defaults to False.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The units the value is written in. None, the default, inherits the scale of
        the field the parameter is passed to, or acts as 1.0.
    metadata : Any, optional
        Arbitrary metadata for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        The bounded parameter.
    """
    return Param(value=value, constraint=Interval(lower, upper), scale=scale, name=name, fixed=fixed, metadata=metadata)


def Random(
    distribution: AbstractDistribution,
    *,
    constraint: Optional[AbstractConstraint] = None,
    value: Optional[ArrayLike] = None, 
    fixed: bool = False,
    name: Optional[str] = None,
    scale: Optional[float] = None,
    metadata: Optional[Any] = None,
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
        The initial declared value. If None, the distribution's mean is used.
    fixed : bool, optional
        Initializes the parameter as fixed. Defaults to False.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The units the value is written in. None, the default, inherits the scale of
        the field the parameter is passed to, or acts as 1.0.
    metadata : Any, optional
        Arbitrary metadata for the parameter, by default None.

    Returns
    -------
    pmrf.Param
        The random parameter.

    Raises
    ------
    ValueError
        If `value` is None and the distribution does not implement `mean()`.
    """
    return Param(value=value, distribution=distribution, constraint=constraint, scale=scale, name=name, fixed=fixed, metadata=metadata)


def is_leaf(x: Any) -> bool:
    """
    Returns if `x` is a boundary for parameter traversal.

    Traversing into a parameter would split it into its raw value and metadata. Also
    stops at Parax's opaque nodes via `parax.constraints.is_leaf`, but not at other
    Parax wrappables, so that parameters nested inside them are still found.
    """
    return is_param(x) or prx.constraints.is_leaf(x)


def _unwraps_to_leaf(x: Any) -> bool:
    """
    Returns if `x` is a node :func:`parax.unwrap` collapses.

    Broader than :func:`is_leaf`, which stops only at parameters. Stopping here gives
    a traversal matching the shape of the unwrapped tree.
    """
    return prx.is_unwrappable(x) or prx.constraints.is_leaf(x)


def node_distribution(node) -> AbstractDistribution | None:
    """
    Returns the prior distribution attached to a parameter, if any.

    A parameter's distribution is authored in declared space, whereas an unwrapped
    tree holds physical values, so the scale is folded into the distribution here. Its Jacobian is constant
    and so cannot move the mode.
    """
    if is_param(node):
        distribution = node.distribution
        if distribution is None:
            return None
        distribution = prx.as_unwrapped(distribution)
        to_physical = node.declared_to_physical_bijector
        if to_physical is not None:
            distribution = Transformed(distribution, to_physical)
        return distribution
    return None


def tree_param_distributions(tree) -> Any:
    """
    Extracts the prior distributions of a tree's parameters.

    The result mirrors the tree once unwrapped, holding each distribution in place of
    the parameter it belongs to and `None` where there is no prior. Distributions are
    metadata and are stripped by unwrapping, so this allows them to be extracted while
    a tree is still wrapped and evaluated against its values afterwards.

    Parameters
    ----------
    tree : PyTree
        The tree to extract from. Must still be wrapped.
    """
    def build(node):
        distribution = node_distribution(node)
        if distribution is not None:
            return distribution
        if prx.is_unwrappable(node):
            # The node vanishes on unwrapping, so mirror what it leaves behind.
            return build(node.unwrap())
        if jax.tree_util.all_leaves([node]):
            return None
        return jax.tree.map(build, node, is_leaf=lambda x: x is not node and _unwraps_to_leaf(x))

    return build(tree)


def tree_param_log_prob(distributions, tree) -> jnp.ndarray:
    """
    Evaluates extracted prior distributions against an unwrapped tree's values.

    Parameters
    ----------
    distributions : PyTree
        The distributions from :func:`tree_param_distributions`.
    tree : PyTree
        The unwrapped tree to evaluate at.
    """
    is_scored = lambda x: x is None or prx.is_distribution(x)
    log_probs = jax.tree.map(
        lambda d, value: d.log_prob(value) if prx.is_distribution(d) else jnp.asarray(0.0),
        distributions, tree, is_leaf=is_scored,
    )
    # An array-valued parameter scores one density per element, so each leaf is reduced
    # before summing; otherwise the result is a vector and the objective stops being
    # scalar.
    return sum(jnp.sum(log_prob) for log_prob in jax.tree.leaves(log_probs))


def _is_name_leaf(x: Any) -> bool:
    """Traversal boundary for name resolution: stops at parameters and Parax's opaque
    nodes, but descends through constants so frozen parameters keep their names."""
    return is_param(x) or (prx.constraints.is_leaf(x) and not prx.is_constant(x))


def _is_name_transparent(x: Any) -> bool:
    """Wrappers whose own path parts are omitted from parameter names."""
    from pmrf.models.adapters.derived import Derived
    from pmrf.models.adapters.wrapped import Wrapped
    from pmrf.modules.base import Module
    from pmrf.modules.wrapped import Tied

    if isinstance(x, (Tied, Wrapped, Derived)):
        return True
    return isinstance(x, prx.AbstractUnwrappable) and not isinstance(x, Module) and not is_param(x)


def _is_frozen_path(tree, path: tuple[Any, ...]) -> bool:
    """Returns whether any node along the JAX key `path` into `tree` is frozen."""
    return any(prx.is_constant(parent) for parent, *_ in path_nodes(tree, path))


def tree_param_paths(tree, free_only: bool = False) -> dict[str, tuple[tuple[Any, ...], Param | jnp.ndarray]]:
    """
    Resolves every parameter name in a tree to its JAX path and node.

    This is the single name resolver behind :func:`params`, :func:`update` and
    :func:`tie`, so a name produced by one is accepted by the others.
    Nested named modules are joined with ``_``.

    Names see through freezing (a frozen parameter keeps its name, but is not free)
    and through Parax wrappers such as :class:`pmrf.modules.Tied`, whose own path
    parts are omitted so that names are relative to the wrapped module. Parax's
    opaque nodes are not descended into, and raw arrays inside frozen sub-trees are
    constant data rather than parameters.

    Parameters
    ----------
    tree : PyTree
        The tree to resolve names in.
    free_only : bool, default=False
        Only resolve free parameters.

    Returns
    -------
    dict[str, tuple[tuple, Param | jax.Array]]
        Names mapped to ``(path, node)``, where ``path`` is a JAX key path into `tree`.

    Raises
    ------
    ValueError
        If two parameters resolve to the same name.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=_is_name_leaf)
    resolved = {}
    for path, leaf in leaves:
        frozen = _is_frozen_path(tree, path)
        if is_param(leaf):
            if free_only and (leaf.fixed or frozen):
                continue
        elif not isinstance(leaf, jax.Array) or frozen:
            continue

        name = path_to_name(tree, path, namespace_separator='_', is_transparent=_is_name_transparent)
        if name in resolved:
            raise ValueError(
                f"Parameter name collision: '{name}'.\n\n"
                f"Multiple paths resolved to the same name during flattening. "
                f"To fix this, either assign unique names directly to the parameters, "
                f"or give their parent models distinct names to create unique prefixes."
            )
        resolved[name] = (path, leaf)
    return resolved


def tree_param_names_to_path(tree) -> dict[str, tuple[Any, ...]]:
    """
    Maps every parameter name in a tree to its JAX path, via :func:`tree_param_paths`.
    """
    return {name: path for name, (path, _) in tree_param_paths(tree).items()}


Space = Literal['raw', 'declared', 'physical']
"""A value space: ``'raw'``, ``'declared'`` or ``'physical'``. See :class:`Param`."""

Selector = Union[str, Sequence[str], Callable[[Any], Any]]
"""A parameter name, an `fnmatch` glob over names, a sequence of them, or a callable
returning the nodes to select, as for :func:`equinox.tree_at`."""


def _check_space(space: str) -> None:
    if space not in ('raw', 'declared', 'physical'):
        raise ValueError(f"Unknown space {space!r}; expected 'raw', 'declared' or 'physical'.")


def _select(tree, where: Selector, free_only: bool = False) -> dict[str, tuple[tuple[Any, ...], Any]]:
    """Resolves the names in `tree` the selector `where` picks, mapped to ``(path, node)``.

    A string or sequence element is an exact name or an `fnmatch` glob. An element
    with no glob characters that names no parameter raises; a glob matching nothing
    selects nothing. A callable returns nodes
    of `tree`; every parameter at or below them is selected.
    """
    resolved = tree_param_paths(tree)
    if callable(where):
        selected = where(tree)
        ids = {id(selected)} | {id(x) for x in jax.tree.leaves(selected, is_leaf=_is_name_leaf)}
        matched = {name for name, (_, leaf) in resolved.items() if id(leaf) in ids}
    else:
        if not _is_selector(where):
            raise TypeError("A selector must be a name, a glob, a sequence of names, or a callable.")
        patterns = [where] if isinstance(where, str) else list(where)
        matched = set()
        for pattern in patterns:
            if pattern in resolved:
                matched.add(pattern)
                continue
            hits = {name for name in resolved if fnmatch.fnmatchcase(name, pattern)}
            if not hits and not any(c in pattern for c in '*?['):
                raise ValueError(f"Unknown parameter name: '{pattern}'")
            matched |= hits
    if free_only:
        matched = {name for name in matched if not _is_fixed_path(tree, *resolved[name])}
    return {name: node for name, node in resolved.items() if name in matched}


def _is_fixed_path(tree, path: tuple[Any, ...], leaf: Any) -> bool:
    """Returns whether the node `leaf` at `path` is not free: fixed, or frozen."""
    return (is_param(leaf) and leaf.fixed) or _is_frozen_path(tree, path)


def _is_selector(x: Any) -> bool:
    """Returns whether `x` is a selector: a name, a sequence of names, or a callable."""
    return callable(x) or isinstance(x, str) or (
        isinstance(x, (list, tuple)) and all(isinstance(s, str) for s in x)
    )


def _read(node, space: str) -> Array:
    """Returns the value of a parameter or a raw array, in `space`."""
    if not is_param(node):
        return jnp.asarray(node)
    if space == 'raw':
        return node.raw_value
    if space == 'physical':
        return node.physical_value
    return node.value


def params(tree, where: Selector = '*', *, free_only: bool = False) -> dict[str, Param]:
    """
    Returns the parameters of a model, or any collection of models, by name.

    Parameters and modules can be given names upon construction. If no custom
    names are present, attribute paths are used. A named module collapses the path
    to its left into a namespace, and a named parameter collapses its path to the
    nearest named module or the root. Nested named modules are joined with ``_``.
    String dictionary keys that are identifiers become dotted names
    (``components.cable.length``); other keys keep the bracket form.

    Names see through freezing and wrappers such as :class:`pmrf.modules.Tied`. The
    target of a tie is derived rather than stored, so it is not named.

    Parameters
    ----------
    tree : PyTree
        A model, or any collection of models and parameters.
    where : str, Sequence[str] or Callable, default='*'
        The parameters to return: a name, an `fnmatch` glob over names, a sequence
        of them, or a callable returning nodes of `tree`. An unknown name raises;
        a glob matching nothing selects nothing.
    free_only : bool, default=False
        Only return free parameters: not fixed, and not frozen.

    Returns
    -------
    dict[str, Param]
        Names mapped to parameters. In a collection that is not a
        :class:`pmrf.Module`, a raw array leaf is returned as it is.

    Raises
    ------
    ValueError
        If two parameters resolve to the same name, or `where` names an unknown one.

    Examples
    --------
    .. code-block:: python

        rc = Resistor(50.0, name='r') ** Capacitor(prf.Unconstrained(2.0, scale=1e-12), name='c')
        prf.params(rc)          # {'r.R': Param(...), 'c.C': Param(...)}
        prf.params(rc, 'c.*')   # {'c.C': Param(...)}
    """
    return {name: leaf for name, (_, leaf) in _select(tree, where, free_only).items()}


def param_values(
    tree,
    where: Selector = '*',
    *,
    free_only: bool = False,
    space: Space = 'declared',
) -> dict[str, Array]:
    """
    Returns the values of a model's parameters by name, in one space.

    This is what :func:`update` accepts, so
    ``prf.update(m, prf.param_values(m, space=s), space=s)`` gives back `m`, with
    the same structure and jit cache key.

    Parameters
    ----------
    tree : PyTree
        A model, or any collection of models and parameters.
    where : str, Sequence[str] or Callable, default='*'
        The parameters to read, as for :func:`params`.
    free_only : bool, default=False
        Only return free parameters.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space of the values. See :class:`Param`.

    Returns
    -------
    dict[str, jax.Array]
        Names, as from :func:`params`, mapped to values.

    Examples
    --------
    .. code-block:: python

        c = Capacitor(prf.Unconstrained(2.0, scale=1e-12))
        prf.param_values(c)                      # {'C': 2.0}
        prf.param_values(c, space='physical')    # {'C': 2e-12}
    """
    _check_space(space)
    return {name: _read(leaf, space) for name, leaf in params(tree, where, free_only=free_only).items()}


def log_prior(tree, *, space: Space = 'declared') -> Array:
    """
    Returns the log prior density of a model's parameters, in one space.

    For parameter values $x$ in declared space with priors $p(x)$, scale $s$ and
    raw values $z$ mapped to declared space by $x = f(z)$:

    $$\\log p_{\\text{declared}} = \\sum_i \\log p_i(x_i)$$

    $$\\log p_{\\text{physical}} = \\log p_{\\text{declared}} - \\sum_i n_i \\log |s_i|$$

    $$\\log p_{\\text{raw}} = \\log p_{\\text{declared}} + \\sum_i \\log \\left|\\det \\frac{\\partial f_i}{\\partial z_i}\\right|$$

    where $n_i$ is the number of elements in parameter $i$. The scale and Jacobian
    terms are the change-of-variables formula for densities. Parameters without a
    prior add nothing to the first two sums (a flat prior). The scale term covers
    parameters with a prior; the Jacobian term covers every free, constrained
    parameter, since those are the coordinates an optimiser or sampler moves. The
    priors of fixed and frozen parameters are included as constants.

    Parameters
    ----------
    tree : PyTree
        A model, or any collection of models and parameters. Must still be wrapped.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space the density is over.

    Returns
    -------
    jax.Array
        A scalar.

    References
    ----------
    .. [1] G. Casella and R. L. Berger, *Statistical Inference*, 2nd ed., Duxbury,
       2002, Theorem 2.1.5 (univariate) and Section 4.3 (multivariate
       transformations).
    """
    _check_space(space)
    # Priors are extracted in physical space, where an unwrapped tree lives.
    physical = tree_param_log_prob(tree_param_distributions(tree), prx.unwrap(tree))
    if space == 'physical':
        return physical

    resolved = tree_param_paths(tree)
    declared = physical + _log_scale(tree)
    if space == 'declared':
        return declared

    free = [p for path, p in resolved.values() if not _is_fixed_path(tree, path, p)]
    log_det = sum(
        (jnp.sum(_raw_log_det_jacobian(p)) for p in free),
        start=jnp.asarray(0.0),
    )
    return declared + log_det


def _log_scale(tree) -> Array:
    """Returns the sum of n log|scale| over the scaled parameters of `tree` with a prior:
    the constant taking the physical log prior to the declared one."""
    return sum(
        (jnp.size(p.value) * jnp.log(jnp.abs(p._scale)) for _, p in tree_param_paths(tree).values()
         if is_param(p) and p.distribution is not None and p._scale != 1.0),
        start=jnp.asarray(0.0),
    )


def _raw_log_det_jacobian(node) -> Array:
    """Returns log|det J| of the map from a node's raw value to its declared value."""
    if is_param(node) and node.constraint is not None:
        return node.raw_to_declared_bijector.forward_log_det_jacobian(node.raw_value)
    return jnp.asarray(0.0)


def _set_paths(tree, paths: list, nodes: list):
    """Replaces the nodes at several JAX key paths at once."""
    if not paths:
        return tree
    getter = Pathgetter(*paths)
    return eqx.tree_at(getter, tree, nodes[0] if len(paths) == 1 else tuple(nodes))


def _write(node, value: Any, space: str):
    """Returns `node`, a parameter or raw array, with its value in `space` replaced.

    A parameter goes through its constructor, keeping everything but the value and
    the jit cache key. A `Param` passed as `value` gives its own value in `space`.
    """
    if is_param(value):
        value = _read(value, space)
    if not is_param(node):
        return _like(jnp.asarray(value, dtype=node.dtype), node)
    if space == 'raw':
        inner = _peel_fixed(node.variable)
        new_inner = eqx.tree_at(lambda v: v.raw_value, inner, _like(jnp.asarray(value), inner.raw_value))
        variable = prx.Fixed(new_inner) if node.fixed else new_inner
        return dataclasses.replace(node, variable=variable)
    if space == 'physical' and node._scale != 1.0:
        value = jnp.asarray(value) / node._scale
    return dataclasses.replace(node, value=value)


def _with_fixed(node, fixed: bool):
    """Returns `node`, a parameter or raw array, with its fixed state set."""
    if not is_param(node):
        return Param(value=node, fixed=True) if fixed else node
    if node.fixed == fixed:
        return node
    variable = prx.Fixed(node.variable) if fixed else prx.as_free(node.variable)
    return dataclasses.replace(node, variable=variable)


def _tree_submodel_paths(tree) -> dict[str, list[tuple[Any, ...]]]:
    """Names every sub-model below the root of `tree`, mapped to the paths with that name.

    Names follow :func:`tree_param_paths`, so a sub-model is named like the prefix of its
    parameters' names (``cascade[1]``, ``load``). A wrapper and the module it wraps share
    a name; the outermost is kept. Two unrelated sub-models with one name both appear,
    and selecting that name raises.
    """
    from pmrf.modules.base import Module

    found: dict[str, list[tuple[Any, ...]]] = {}

    def walk(node, prefix):
        is_leaf = lambda x: x is not node and (isinstance(x, Module) or _is_name_leaf(x))
        for path, leaf in jax.tree_util.tree_flatten_with_path(node, is_leaf=is_leaf)[0]:
            if not isinstance(leaf, Module):
                continue
            full = prefix + tuple(path)
            name = path_to_name(tree, full, namespace_separator='_', is_transparent=_is_name_transparent)
            if name:
                paths = found.setdefault(name, [])
                if not any(full[:len(p)] == p for p in paths):
                    paths.append(full)
            walk(leaf, full)

    walk(tree, ())
    return found


def _submodel_path(name: str, submodels: dict[str, list[tuple[Any, ...]]]) -> tuple[Any, ...]:
    """The path of the one sub-model called `name`, as `_tree_submodel_paths` names them.

    A name two unrelated sub-models share picks neither, so it raises.
    """
    if len(submodels[name]) > 1:
        raise ValueError(f"Sub-model name '{name}' is ambiguous: several sub-models have it.")
    return submodels[name][0]


def _select_parts(tree, where: Selector) -> list[tuple[Any, ...]]:
    """Resolves the paths of the parts a string or sequence selector picks, for the
    structural forms of :func:`update`.

    An exact name selects a parameter, or failing that a sub-model. Anything else is
    an `fnmatch` glob over parameter names.
    """
    if not _is_selector(where) or callable(where):
        raise TypeError("A selector must be a name, a glob, a sequence of names, or a callable.")
    resolved = tree_param_paths(tree)
    submodels = None
    paths = []
    for pattern in ([where] if isinstance(where, str) else list(where)):
        if pattern in resolved:
            paths.append(resolved[pattern][0])
            continue
        if submodels is None:
            submodels = _tree_submodel_paths(tree)
        if pattern in submodels:
            paths.append(_submodel_path(pattern, submodels))
            continue
        hits = [name for name in resolved if fnmatch.fnmatchcase(name, pattern)]
        if not hits and not any(c in pattern for c in '*?['):
            raise ValueError(f"Unknown parameter or sub-model name: '{pattern}'")
        paths.extend(resolved[name][0] for name in hits)

    unique = list(dict.fromkeys(paths))
    _check_no_overlap(unique)
    return unique


def _check_no_overlap(paths: list[tuple[Any, ...]]):
    """Raises if one of the selected paths contains another."""
    for a in paths:
        for b in paths:
            if a != b and b[:len(a)] == a:
                raise ValueError("The selected parts overlap: one contains another.")


_UPDATE_FORMS = """prf.update takes one of these forms:
    update(model, {'name': value, ...}, space=...)   values by name, or sub-models by name
    update(model, where, value=..., space=...)       one value for the selected parameters
    update(model, where, fixed=True or False)        fixed state of the selected parameters
    update(model, where, node)                       replace the selected parts with `node`
    update(model, where, fn=...)                     replace each selected part with fn(old)
    update(param, value=..., space=...)              the parameter itself
    update(param, fixed=True or False)
where `where` is a name, a glob, a sequence of names, or a callable."""

_MISSING = object()


def update(
    tree,
    selection: Any = _MISSING,
    node: Any = _MISSING,
    /,
    *,
    value: Any = _MISSING,
    fixed: bool | None = None,
    space: Space | None = None,
    fn: Callable[[Any], Any] | None = None,
):
    """
    Returns a copy of a model with the parts a selector picks replaced.

    Exactly one form says what replaces them:

    .. code-block:: python

        prf.update(model, {'L1.L': 3.0, 'C1.C': 2.0})   # values by name
        prf.update(model, {'load': Short(), 'L1.L': 3.0})  # sub-models and values by name
        prf.update(model, 'L1.*', value=3.0)            # one value for a selection
        prf.update(model, 'cable.*', fixed=True)        # fixed state
        prf.update(model, 'cascade[1]', Short())        # a new sub-model or node
        prf.update(model, 'load.*', fn=lambda p: ...)   # a function of the old part
        prf.update(model, v, space='raw')               # write-back from an optimiser or sampler
        prf.update(param, value=3.0)                    # the parameter itself

    The mapping, `value` and `fixed` forms go through each parameter's constructor:
    values are checked against the bounds, and the prior, constraint, scale, name and
    metadata are kept. Value forms keep the model's structure, and every leaf's dtype,
    shape and `weak_type`, so RF methods such as :meth:`pmrf.Model.s` do not recompile.
    Changing `fixed` changes the structure, and recompiling is expected.

    The `node` and `fn` forms are structural: they bypass converters and validation,
    and put exactly what they are given in place of each selected part, so the caller
    keeps field invariants. They usually change the structure, and recompile.

    In the mapping form the tier is decided per entry by the value's type. A
    :class:`pmrf.Model` value is a structural replacement of the sub-model its key
    names, as in ``update(model, key, node)``: it is unvalidated and usually
    recompiles, and `space` does not apply to it. Any other value is a validated
    value update of the parameter its key names. A mapping of values only keeps the
    jit cache key.

    ``fixed=`` is additive: parameters the selector does not match are untouched,
    and ``fixed=False`` frees a parameter even if it was created fixed. It does not
    unfreeze: a parameter inside a :func:`pmrf.freeze` sub-tree stays frozen. To
    leave only some parameters free, fix everything, then free those:
    ``update(update(m, '*', fixed=True), names, fixed=False)``.

    `update` is not an optimiser step. For replacing a field of one object without
    validation, use :func:`pmrf.replace`. To derive one parameter from another, use
    :func:`tie`.

    Parameters
    ----------
    tree : PyTree
        A model, any collection of models and parameters, or a single parameter.
    selection : Mapping[str, ArrayLike | Param] or selector, optional
        Either a mapping from names to values, or a selector: a name, an `fnmatch`
        glob over names, a sequence of them, or a callable returning nodes of
        `tree`. A mapping is recognised only when every key is a string. Its values
        may be arrays, or parameters, whose value in `space` is used, keyed by
        parameter name; or models, keyed by sub-model name, which replace that
        sub-model structurally. Omit it to
        update `tree` itself, which must then be a parameter. In the structural
        forms, an exact name may also name a sub-model (``'cascade[1]'``, or a named
        module's name), a glob matches parameter names only, and a callable selects
        the nodes it returns rather than the parameters below them.
    node : Any, optional
        The part to put in place of each selected part.
    value : ArrayLike, optional
        The value to give every selected parameter.
    fixed : bool, optional
        The fixed state to give every selected parameter.
    space : {'declared', 'physical', 'raw'}, optional
        The space of the values, by default ``'declared'``. Only used with values.
    fn : Callable, optional
        Called on each selected part; its result replaces the part.

    Returns
    -------
    PyTree
        The updated copy.

    Raises
    ------
    ValueError
        If a name is unknown, a value is outside its parameter's constraint, or
        structurally selected parts overlap. Under `jax.jit` the bounds check raises
        at runtime.
    TypeError
        If the arguments match none of the forms, or a mapping gives a model for a
        parameter name or a non-model for a sub-model name.
    """
    has_value = value is not _MISSING
    has_fixed = fixed is not None
    has_node = node is not _MISSING
    has_fn = fn is not None

    def form_error():
        return TypeError(f"prf.update got arguments matching none of its forms.\n\n{_UPDATE_FORMS}")

    if has_value + has_fixed + has_node + has_fn > 1:
        raise form_error()
    if space is not None and not has_value and not isinstance(selection, Mapping):
        raise TypeError(f"`space` applies only to values.\n\n{_UPDATE_FORMS}")
    space = 'declared' if space is None else space
    _check_space(space)

    if has_node or has_fn:
        if selection is _MISSING or isinstance(selection, Mapping) or not _is_selector(selection):
            raise form_error()
        replace_fn = fn if has_fn else (lambda _: node)
        if callable(selection):
            return eqx.tree_at(selection, tree, replace_fn=replace_fn)
        paths = _select_parts(tree, selection)
        if not paths:
            return tree
        parts = [Pathgetter(path)(tree) for path in paths]
        return _set_paths(tree, paths, [replace_fn(part) for part in parts])

    if selection is _MISSING:
        if not is_param(tree) or has_value == has_fixed:
            raise form_error()
        return _write(tree, value, space) if has_value else _with_fixed(tree, fixed)

    if isinstance(selection, Mapping):
        if has_value or has_fixed or not all(isinstance(k, str) for k in selection):
            raise form_error()
        from pmrf.models.base import Model

        resolved = tree_param_paths(tree)
        submodels = None
        paths, nodes, unknown = [], [], []
        for name, v in selection.items():
            if isinstance(v, Model):
                if name in resolved:
                    raise TypeError(f"'{name}' is a parameter name, but its value is a model; "
                                    "a model can only replace a sub-model.")
                if submodels is None:
                    submodels = _tree_submodel_paths(tree)
                if name not in submodels:
                    raise ValueError(f"Unknown sub-model name: '{name}'")
                paths.append(_submodel_path(name, submodels))
                nodes.append(v)
            elif name in resolved:
                paths.append(resolved[name][0])
                nodes.append(_write(resolved[name][1], v, space))
            else:
                if submodels is None:
                    submodels = _tree_submodel_paths(tree)
                if name in submodels:
                    raise TypeError(f"'{name}' names a sub-model, but its value is not a model; "
                                    "only a pmrf.Model can replace a sub-model.")
                unknown.append(name)
        if unknown:
            raise ValueError(f"Unknown parameter names: {unknown}")
        if submodels is not None:
            _check_no_overlap(paths)
        return _set_paths(tree, paths, nodes)

    if not _is_selector(selection) or has_value == has_fixed:
        raise form_error()
    selected = _select(tree, selection)
    paths = [path for path, _ in selected.values()]
    if has_value:
        nodes = [_write(leaf, value, space) for _, leaf in selected.values()]
    else:
        nodes = [_with_fixed(leaf, fixed) for _, leaf in selected.values()]
    return _set_paths(tree, paths, nodes)


def _identity(x):
    return x


def _submodel_parameters(name: str, path: tuple[Any, ...], resolved: dict) -> dict[str, str]:
    """Maps each suffix below the sub-model `name` to the parameter it names.

    A parameter below the sub-model is named after it, so the rest of its name is the
    suffix the two sides of a tie are paired on.
    """
    below = {}
    for parameter, (parameter_path, _) in resolved.items():
        if parameter_path[:len(path)] != path:
            continue
        if not parameter.startswith(name):
            raise ValueError(
                f"Parameter '{parameter}' lies below sub-model '{name}' but is not named "
                f"after it, so there is no suffix to pair it on."
            )
        below[parameter[len(name):]] = parameter
    if not below:
        raise ValueError(f"Sub-model '{name}' has no parameters to tie.")
    return below


def _tie_pairs(tree, resolved: dict, target: Selector, source: Selector) -> list[tuple[Selector, Selector]]:
    """The target and source pairs one call to :func:`tie` stands for.

    A name resolves as it does for :func:`update`: an exact name selects a parameter,
    or failing that a sub-model. Two sub-model names expand to the parameters beneath
    them, paired by the suffix below each name, and a suffix on one side only raises
    rather than tying a partial set. Anything else — a leaf name, a sequence of names,
    a callable — is one pair, passed through as it was given.
    """
    if not (isinstance(target, str) and isinstance(source, str)):
        return [(target, source)]
    if target in resolved and source in resolved:
        return [(target, source)]

    submodels = _tree_submodel_paths(tree)
    is_submodel = [name in submodels and name not in resolved for name in (target, source)]
    if not any(is_submodel):
        return [(target, source)]  # `resolve_target` reports a name it cannot find.
    unknown = [name for name in (target, source) if name not in resolved and name not in submodels]
    if unknown:
        raise ValueError(f"Unknown parameter or sub-model name: '{unknown[0]}'")
    if not all(is_submodel):
        submodel, parameter = (target, source) if is_submodel[0] else (source, target)
        raise ValueError(
            f"'{submodel}' names a sub-model and '{parameter}' a parameter, so there is "
            f"no suffix to pair them on: tie two sub-models, or two parameters."
        )

    targets = _submodel_parameters(target, _submodel_path(target, submodels), resolved)
    sources = _submodel_parameters(source, _submodel_path(source, submodels), resolved)
    unpaired = [targets[s] for s in targets.keys() - sources.keys()]
    unpaired += [sources[s] for s in sources.keys() - targets.keys()]
    if unpaired:
        raise ValueError(
            f"'{target}' and '{source}' do not have matching parameter names beneath "
            f"them, so tying them would tie a partial set. Unpaired: "
            f"{', '.join(repr(name) for name in sorted(unpaired))}."
        )
    return [(targets[suffix], sources[suffix]) for suffix in sorted(targets)]


def tie(tree, target: Selector, source: Selector, fn: Callable[[Any], Any] = _identity):
    """
    Returns a copy of a model in which one part is derived from another.

    The target is removed from the model's parameters and recomputed as
    ``fn(source)`` every time the tree is resolved, so it follows the source through
    :func:`update`, optimisation and sampling. A tie is not a replacement: use
    :func:`update` to replace a part once.

    ``fn`` receives the source as the tie is applied, so a parameter arrives as
    its physical value. Tying a model that is already tied adds a tie; names refer
    to the untied model.

    A name resolves as it does for :func:`update`: an exact name selects a parameter,
    or failing that a sub-model. A sub-model name, on either side, ties every
    parameter beneath it, pairing target and source by the suffix below the name and
    applying `fn` to each pair.

    Parameters
    ----------
    tree : PyTree
        A model, or any collection of models and parameters.
    target : str, Sequence[str] or Callable
        The part to derive: a parameter name, a sub-model name, a sequence of names,
        or a callable returning nodes of `tree`.
    source : str, Sequence[str] or Callable
        The part it is derived from, selected the same way.
    fn : Callable, optional
        Maps the source to the target. Defaults to the identity.

    Returns
    -------
    PyTree
        A :class:`pmrf.models.Wrapped` if `tree` is a :class:`pmrf.Model`, so RF
        methods stay available; otherwise a :class:`pmrf.modules.Tied`, which is not
        a container. Read either back with :func:`resolve`, which applies the ties
        and returns the tree in its own shape with its parameters intact.

    See Also
    --------
    resolve : Apply a tree's ties, leaving its parameters as parameters.

    Raises
    ------
    ValueError
        If a name is not found, if a sub-model name is ambiguous, or if the two
        sub-models do not have matching parameter names beneath them.

    Examples
    --------
    .. code-block:: python

        rc = Resistor(50.0, name='r') ** Capacitor(1e-12, name='c')
        tied = prf.tie(rc, 'r.R', 'c.C', fn=lambda C: C * 5e13)
        prf.params(tied)                                     # {'c.C': Param(...)}
        prf.update(tied, {'c.C': 2e-12}).build().cascade[0].R   # 100.0

        parts = {'strip_a': MicrostripLine(...), 'strip_b': MicrostripLine(...)}
        tied = prf.tie(parts, 'strip_b', 'strip_a')          # every parameter beneath
        prf.params(tied)                                     # only `strip_a`'s
    """
    from pmrf.models import Model, Wrapped
    from pmrf.modules import Tied

    base = tree.wrapped if isinstance(tree, Wrapped) else tree
    untied = base.module if isinstance(base, Tied) else base
    resolved = tree_param_paths(untied)
    name_to_path = {name: path for name, (path, _) in resolved.items()}
    tied = base
    for one_target, one_source in _tie_pairs(untied, resolved, target, source):
        tied = Tied(
            tied,
            target=resolve_target(one_target, name_to_path),
            source=resolve_target(one_source, name_to_path),
            tie_fn=fn,
        )
    return Wrapped(wrapped=tied) if isinstance(tree, Model) else tied


def _is_tie(x: Any) -> bool:
    """Returns whether `x` is a node a tie is made of.

    Both classes are needed: :meth:`pmrf.modules.Tied.unwrap` hands back the inner
    :class:`parax.Tie` rather than a resolved tree, so discharging one tie takes two
    steps across two classes, and a predicate matching only `Tied` leaves a bare
    `parax.Tie` behind.
    """
    from pmrf.modules import Tied

    return isinstance(x, (Tied, prx.Tie))


def resolve(tree):
    """
    Returns a copy of a tree with its ties applied and its parameters left intact.

    Resolution is structural: every tie made by :func:`tie` is discharged, and
    everything else the tree holds is left as it stands. This is the counterpart of
    :func:`pmrf.unwrap`, which is evaluation: unwrapping replaces every parameter
    with its physical value, so nothing rebuilt from its result is parameterised.
    Use `resolve` to read a tied tree back, and `unwrap` only to evaluate one.

    A tie's target resolves to a plain value either way, since it is derived and has
    no prior of its own. Every other parameter survives as a parameter.

    Parameters
    ----------
    tree : PyTree
        A model, or any collection of models and parameters.

    Returns
    -------
    PyTree
        The tree in its own shape: a `dict` of components resolves to a `dict`, a
        `list` to a `list`, and a :class:`pmrf.Model` to a model whose RF methods
        work. A tree with no ties comes back unchanged.

    See Also
    --------
    tie : Derive one part of a tree from another.

    Examples
    --------
    .. code-block:: python

        parts = {'a': Resistor(prf.Random(...), name='a'), 'b': Resistor(50.0, name='b')}
        tied = prf.tie(parts, 'b.R', 'a.R')
        resolved = prf.resolve(tied)
        resolved['b'].R                  # the tied value
        prf.is_param(resolved['a'].R)    # True: untied parameters survive
    """
    return prx.unwrap(tree, only_if=_is_tie, cascade=False)


__all__ = [
    "Param",
    "is_param",
    "as_param",
    "param",
    "Fixed",
    "Unconstrained",
    "Bounded",
    "Constrained",
    "Random",
    "params",
    "param_values",
    "log_prior",
    "update",
    "tie",
    "resolve",
]
