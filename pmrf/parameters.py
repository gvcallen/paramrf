"""
Parameter factories, converters, and field specifiers.

Most of these are re-exported at root.

Builds on top of `Parax <https://gvcallen.github.io/parax>`_.
"""
from __future__ import annotations

import dataclasses
import fnmatch
from typing import Any, Literal, Optional, Self, Sequence, Union, Callable, TypeVar, TypeGuard

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, Array
import equinox as eqx
import parax as prx
from parax.annotation import AbstractAnnotated

from pmrf.bijectors import AbstractBijector, Chain, ScalarAffine
from pmrf.constraints import AbstractConstraint, Interval
from pmrf.distributions import AbstractDistribution, Transformed
from pmrf.utils import error_if, field
from pmrf.utils.optix import focus, Lens
from pmrf.utils.tree import path_nodes, path_to_name


T = TypeVar('T')


class Param(prx.AbstractVariable, prx.AbstractWrappable[Array], AbstractAnnotated[Any]):
    """
    The canonical parameter container for ParamRF.

    Parameters can be created by instantiating this class, or using factories
    in :mod:`pmrf.parameters`, most of which are re-exported at root
    (e.g. :func:`pmrf.Unconstrained`, :func:`pmrf.Fixed`, :func:`pmrf.Bounded`).
    
    Wraps a `Parax <https://gvcallen.github.io/parax>`_ variable,
    applying an optional scale, name and metadata.
    """
    #: The raw value of the parameter.
    raw_value: prx.AbstractVariable = eqx.field(converter=prx.as_variable)

    #: The scale of the parameter.
    scale: float = eqx.field(converter=float, default=1.0, static=True)
    
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
        scale: float = 1.0,
        fixed: bool = False,
        metadata: Any = None,
        raw_value: Optional[prx.AbstractVariable] = None,
    ):
        """
        Creates a generic parameter.

        The incoming value can be any ArrayLike object.

        Parameters
        ----------
        value : ArrayLike, optional
            The unscaled value of the parameter.
        distribution : Optional[AbstractDistribution], optional
            The unscaled probability distribution for the parameter. See :mod:`pmrf.distributions`.
        constraint : Optional[AbstractConstraint], optional
            The unscaled constraint to apply to the parameter. See :mod:`pmrf.constraints`.
        name : str, optional
            A name for the parameter, by default None.
        scale : float, optional
            The scaling factor to apply, by default 1.0.
        fixed : bool, optional
            Initializes the parameter as fixed. Defaults to False.
        metadata : Any, optional
            Arbitrary metadata for the parameter, by default None.
        raw_value : Optional[prx.AbstractVariable], optional
            The raw Parax variable to wrap. If `value` is also passed, it is instead
            the physical (scaled) value, as :attr:`value` returns, and the variable's
            value is replaced, keeping its distribution, constraint and fixed state.
            This is what makes ``pmrf.replace(param, value=...)`` work, so
            ``pmrf.replace(param, value=param.value)`` is the identity (up to the
            floating-point round trip through the constraint bijector).
        """
        if isinstance(value, prx.AbstractVariable):
            raise ValueError("Got a Parax variable when constructing a parameter")

        if raw_value is not None and (distribution is not None or constraint is not None):
            raise ValueError("Cannot pass `raw_value` with `distribution` or `constraint` to Param constructor")

        if raw_value is not None and value is not None:
            raw_value = _replace_raw_value(raw_value, jnp.asarray(value) / scale)
        elif raw_value is None:
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
                raw_value = prx.Random(distribution, constraint=constraint, value=value)
            elif constraint is not None:
                raw_value = prx.Constrained(constraint, value=value)
            else:
                raw_value = prx.Real(value)
            if fixed:
                raw_value = prx.Fixed(raw_value)

        self.raw_value = raw_value
        self.scale = float(scale)
        self.name = name
        self.metadata = metadata

    def at(
        self: Self, 
        where: Union[Callable[[Self], T], str, tuple[str, ...], list[str]]
    ) -> Lens[Self, T]:
        """(experimental) A functional interface for parameter manipulation.
        
        This is a wrapper around `equinox.tree_at` via the `jax-optix` library.

        Similar to :meth:`pmrf.Model.at` but only accepts callables.
        See the documentation for that method for more details.

        Returns
        -------
        Lens
            A lens object focused on the root of the current instance.
        """
        return focus(self).at(where)
    
    @property
    def fixed(self) -> bool:
        """
        Indicates whether the parameter is fixed (constant).

        Returns
        -------
        bool
            True if the parameter is fixed, False otherwise.
        """
        return prx.is_constant(self.raw_value)
    
    def as_fixed(self) -> Param:
        """
        Returns a fixed version of this parameter.

        Returns
        -------
        Param
            A new parameter instance wrapped as fixed.
        """
        if self.fixed:
            return self
        return dataclasses.replace(self, raw_value=prx.Fixed(self.raw_value))
    
    def as_free(self) -> Param:
        """
        Returns a free (variable) version of this parameter.

        Returns
        -------
        Param
            A new parameter instance wrapped as free.
        """
        if not self.fixed:
            return self
        return dataclasses.replace(self, raw_value=prx.as_free(self.raw_value))
    
    @property
    def distribution(self) -> AbstractDistribution | None:
        """
        The unscaled probability distribution associated with the parameter.

        Returns
        -------
        AbstractDistribution | None
            The distribution if one exists, otherwise None.
        """
        if prx.is_probabilistic(self.raw_value):
            return self.raw_value.distribution
        return None
    
    @property
    def constraint(self) -> AbstractConstraint | None:
        """
        The unscaled constraint associated with the parameter.

        Returns
        -------
        AbstractConstraint | None
            The constraint if one exists, otherwise None.
        """
        if prx.is_constrained(self.raw_value):
            return self.raw_value.constraint
        return None
    
    @property
    def bounds(self) -> tuple[ArrayLike, ArrayLike] | None:
        """
        The unscaled lower and upper bounds of the parameter.

        Returns
        -------
        tuple[ArrayLike, ArrayLike] | None
            A tuple of (lower_bound, upper_bound) if bounds exist, otherwise None.
        """
        if prx.is_bounded(self.raw_value):
            return self.raw_value.bounds
        return None

    @property
    def raw_to_constrained_bijector(self) -> AbstractBijector | None:
        """
        The bijector mapping the raw value to the constrained value.

        The raw value is the latent one held in `raw_value`, which Parax refers to as
        the unconstrained space. It is called raw here to avoid confusion with
        :func:`pmrf.Unconstrained`, which creates a parameter without bounds.

        Returns
        -------
        AbstractBijector | None
            The bijector if a constraint exists, otherwise None.
        """
        if self.constraint is None:
            return None
        return prx.as_unwrapped(self.constraint).bijector

    @property
    def constrained_to_physical_bijector(self) -> AbstractBijector | None:
        """
        The bijector mapping the constrained value to the scaled physical value.

        This is the parameter's scale. Distributions and bounds are authored in the
        constrained space, so this is the step needed to compare them against a value
        that has been unwrapped.

        Returns
        -------
        AbstractBijector | None
            The bijector if the parameter is scaled, otherwise None.
        """
        if self.scale == 1.0:
            return None
        return ScalarAffine(shift=jnp.array(0.0), scale=jnp.array(self.scale))

    @property
    def bijector(self) -> AbstractBijector | None:
        """
        The full bijector mapping the raw value to the scaled physical value.

        Composes :attr:`raw_to_constrained_bijector` with
        :attr:`constrained_to_physical_bijector`.

        Returns
        -------
        AbstractBijector | None
            The bijector if a constraint exists, otherwise None.
        """
        raw_to_constrained = self.raw_to_constrained_bijector
        if raw_to_constrained is None:
            return None
        constrained_to_physical = self.constrained_to_physical_bijector
        if constrained_to_physical is None:
            return raw_to_constrained
        return Chain([constrained_to_physical, raw_to_constrained])

    @property
    def value(self) -> jax.Array:
        """
        Returns the scaled physical value.

        Returns
        -------
        jax.Array
            The computed array value.
        """
        base_value = jnp.array(self.raw_value)
        if self.scale != 1.0:
            return base_value * self.scale
        return base_value

    @property
    def unscaled_value(self) -> jax.Array:
        """
        Returns the original unscaled value.

        Returns
        -------
        jax.Array
            The computed array value.
        """
        return jnp.array(self.raw_value)

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

    def wrap(self, value: Array) -> Self:
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
        if not prx.is_wrappable(self.raw_value):
            raise ValueError("Cannot wrap a parameter that wraps a non-wrappable Parax variable")
        
        new_raw_value = self.raw_value.wrap(value / self.scale)
        return eqx.tree_at(lambda x: x.raw_value, self, new_raw_value)
    

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


def _replace_raw_value(raw_value: prx.AbstractVariable, value: ArrayLike) -> prx.AbstractVariable:
    """Replaces the unscaled value of a Parax variable, keeping its structure.

    `Fixed` is peeled off before wrapping, since wrapping through it drops the wrapped
    variable's distribution, and the value is checked against the constraint because
    wrapping does not check bounds.
    """
    fixed = isinstance(raw_value, prx.Fixed)
    inner = raw_value.raw_value if fixed else raw_value
    value_array = jnp.asarray(value)
    if prx.is_constrained(inner):
        value_array = _check_in_constraint(value_array, prx.unwrap(inner.constraint))
    inner = inner.wrap(value_array)
    return prx.Fixed(inner) if fixed else inner


def is_param(x: Any) -> TypeGuard[Param]:
    """
    Returns if `x` is an instance of :class:`pmrf.Param`.
    """
    return isinstance(x, Param)


def as_param(
    value: Any = None,
    *,
    constraint: Optional[AbstractConstraint] = None,
    scale: float = 1.0,
    as_free: bool = False,
    as_fixed: bool = False,
) -> Param:
    """
    Coerces a value into a parameter.

    The incoming value can be an existing parameter or parax variable,
    or any parameter-like object (float, array etc.).
    
    Any scaling and constraints are automatically intersected.

    Parameters
    ----------
    value : Any, optional
        The value of the parameter.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter. See :mod:`pmrf.constraints`.
    scale : float, optional
        The scaling factor to apply, by default 1.0.        
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
        scale = value.scale * scale
        name = value.name
        metadata = value.metadata
        value = value.raw_value

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
        p = p.as_fixed()
    if as_free:
        p = p.as_free()
    return p
    

def param(
    *,
    default: Any = dataclasses.MISSING,
    as_free: bool = False,
    as_fixed: bool = False,
    constraint: Optional[AbstractConstraint] = None,
    scale: float = 1.0,
    **kwargs,
) -> Any:
    """
    A field specifier for registering parameters within a model.

    This specifier can be used when declaring custom models inheriting from `pmrf.Model`.

    It is used to register the parameter when a model is constructed, so it is listed
    under :meth:`pmrf.Module.named_params`. It can also be used to enforce
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

        RC(1.0, 2.0)
        # RC(R=1., C=2.e-12)

        RC(-1.0, 2.0)
        # ValueError: out of bounds

    Parameters
    ----------
    default : Any, optional
        The default value of the parameter.
    constraint : Optional[AbstractConstraint], optional
        The constraint to apply to the parameter. See :mod:`pmrf.constraints`.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
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
    scale: float = 1.0,
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
        The initial unscaled parameter value.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
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
    scale: float = 1.0,
    name: Optional[str] = None,
    metadata: Optional[Any] = None,
) -> Param:
    """
    Create an unconstrained free parameter.

    Parameters
    ----------
    value : ArrayLike
        The initial unscaled parameter value.
    fixed : bool, optional
        Wraps the parameter in a :class:`pmrf.Fixed` parameter.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
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
    scale: float = 1.0, 
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
        The initial unscaled value of the parameter.
    fixed : bool, optional
        Initializes the parameter as fixed. Defaults to False.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
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
    scale: float = 1.0, 
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
        The initial unscaled value. If None, the midpoint of the bounds is used.
    fixed : bool, optional
        Initializes the parameter as fixed. Defaults to False.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
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
    scale: float = 1.0, 
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
        The initial unscaled value. If None, the distribution's mean is used.
    fixed : bool, optional
        Initializes the parameter as fixed. Defaults to False.
    name : str, optional
        A name for the parameter, by default None.
    scale : float, optional
        The scaling factor to apply, by default 1.0.
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
    Returns the prior distribution attached to a node, if any.

    Covers both a :class:`Param`'s own distribution and a joint distribution attached
    over a sub-tree, such as by :class:`pmrf.modules.Probabilistic`.

    A parameter's distribution is authored in its raw, unscaled space, whereas an
    unwrapped tree holds scaled values, so the scale is folded into the distribution
    here. Its Jacobian is constant and so cannot move the mode.
    """
    if is_param(node):
        distribution = node.distribution
        if distribution is None:
            return None
        distribution = prx.as_unwrapped(distribution)
        to_physical = node.constrained_to_physical_bijector
        if to_physical is not None:
            distribution = Transformed(distribution, to_physical)
        return distribution
    if prx.is_probabilistic(node):
        return prx.as_unwrapped(node.distribution)
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
            # Covers the whole sub-tree it unwraps to, so a distribution attached
            # higher up overrides any below it.
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
    from pmrf.models.adapters.wrapped import Wrapped
    from pmrf.modules.base import Module
    from pmrf.modules.wrapped import Probabilistic, Tied

    if isinstance(x, (Tied, Probabilistic, Wrapped)):
        return True
    return isinstance(x, prx.AbstractUnwrappable) and not isinstance(x, Module) and not is_param(x)


def _is_frozen_path(tree, path: tuple[Any, ...]) -> bool:
    """Returns whether any node along the JAX key `path` into `tree` is frozen."""
    return any(prx.is_constant(parent) for parent, *_ in path_nodes(tree, path))


def tree_param_paths(
    tree,
    free_only: bool = False,
    namespace_separator: str = '_',
) -> dict[str, tuple[tuple[Any, ...], Param | jnp.ndarray]]:
    """
    Resolves every parameter name in a tree to its JAX path and node.

    This is the single name resolver behind :meth:`pmrf.Module.named_params`,
    :meth:`pmrf.Module.at` and :meth:`pmrf.Module.tied`, so a name produced by one
    is accepted by the others.

    Names see through freezing (a frozen parameter keeps its name, but is not free)
    and through Parax wrappers such as :class:`pmrf.modules.Tied`, whose own path
    parts are omitted so that names are relative to the wrapped module. Parax's
    opaque nodes (e.g. a `parax.Probabilize` target) are not descended into, and raw
    arrays inside frozen sub-trees are constant data rather than parameters.

    Parameters
    ----------
    tree : PyTree
        The tree to resolve names in.
    free_only : bool, default=False
        Only resolve free parameters.
    namespace_separator : str, default='_'
        The separator used to join named module namespaces.

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

        name = path_to_name(
            tree, path, namespace_separator=namespace_separator, is_transparent=_is_name_transparent
        )
        if name in resolved:
            raise ValueError(
                f"Parameter name collision: '{name}'.\n\n"
                f"Multiple paths resolved to the same name during flattening. "
                f"To fix this, either assign unique names directly to the parameters, "
                f"or give their parent models distinct names to create unique prefixes."
            )
        resolved[name] = (path, leaf)
    return resolved


def tree_named_params(
    tree,
    full_params: bool = False,
    free_only: bool = False,
    namespace_separator: str = '_',
) -> dict[str, float | jnp.ndarray | Param]:
    """
    Returns a named dictionary of parameters in a tree.

    Names are resolved by :func:`tree_param_paths`.

    Parameters
    ----------
    full_params : bool, default=False
        Returns the full parameter objects as opposed to their resultant floats/array values.
    free_only : bool, default=False
        Returns only free parameters.
    namespace_separator : str
        The separator to use to create a parameter namespace using model names.

    Returns
    -------
    dict[str, Any]
        Parameter names mapped to their values or parameter objects.
    """
    named = {}
    for name, (_, leaf) in tree_param_paths(
        tree, free_only=free_only, namespace_separator=namespace_separator
    ).items():
        if not full_params:
            leaf = prx.unwrap(leaf)
            if jnp.isscalar(leaf):
                leaf = float(leaf)
        named[name] = leaf
    return named


def tree_param_names_to_path(tree, namespace_separator: str = '_') -> dict[str, tuple[Any, ...]]:
    """
    Maps every parameter name in a tree to its JAX path, via :func:`tree_param_paths`.
    """
    return {
        name: path
        for name, (path, _) in tree_param_paths(tree, namespace_separator=namespace_separator).items()
    }


def tree_param_values(tree, free_only: bool = False) -> dict[str, jnp.ndarray]:
    """
    Returns the physical (scaled) value of every named parameter in a tree.

    Parameters
    ----------
    tree : PyTree
        The tree to read.
    free_only : bool, default=False
        Only return free parameters.

    Returns
    -------
    dict[str, jax.Array]
        Names, as in :func:`tree_named_params`, mapped to physical values.
    """
    return {
        name: leaf.value if is_param(leaf) else jnp.asarray(leaf)
        for name, (_, leaf) in tree_param_paths(tree, free_only=free_only).items()
    }


def _set_paths(tree, paths: list, nodes: list):
    """Replaces the nodes at several JAX key paths at once."""
    if not paths:
        return tree
    from pmrf.utils.tree import Pathgetter
    getter = Pathgetter(*paths)
    return eqx.tree_at(getter, tree, nodes[0] if len(paths) == 1 else tuple(nodes))


def tree_with_values(tree, values: dict[str, ArrayLike], strict: bool = True):
    """
    Returns a tree with parameter values replaced by name.

    The structure is unchanged: each parameter keeps its distribution, constraint,
    scale, name, metadata and fixed or frozen state. ``tree_with_values(tree,
    tree_param_values(tree))`` is the identity, up to the floating-point round trip
    through each parameter's constraint bijector.

    Parameters
    ----------
    tree : PyTree
        The tree to update.
    values : dict[str, ArrayLike]
        Names mapped to physical (scaled) values.
    strict : bool, default=True
        Raise on names that do not resolve. If False, they are ignored.

    Returns
    -------
    PyTree
        The updated tree.

    Raises
    ------
    ValueError
        If `strict` and a name does not resolve, or if a value is outside its
        parameter's constraint. Under `jax.jit`, the bounds check raises at runtime.
    """
    resolved = tree_param_paths(tree)
    unknown = [name for name in values if name not in resolved]
    if strict and unknown:
        raise ValueError(f"Unknown parameter names: {unknown}")

    paths, nodes = [], []
    for name, value in values.items():
        if name not in resolved:
            continue
        path, leaf = resolved[name]
        if is_param(leaf):
            leaf = dataclasses.replace(leaf, value=value)
        else:
            leaf = jnp.asarray(value, dtype=leaf.dtype)
        paths.append(path)
        nodes.append(leaf)
    return _set_paths(tree, paths, nodes)


def _match_names(tree, patterns: str | Sequence[str]):
    """
    Resolves the parameter names in `tree` and selects those matching `patterns`.

    Parameters
    ----------
    tree : PyTree
        The tree to resolve names in.
    patterns : str or Sequence[str]
        `fnmatch` globs, matched case-sensitively against every resolved name.

    Returns
    -------
    resolved : dict[str, tuple[tuple, Param | jax.Array]]
        Every name mapped to ``(path, node)``, as from :func:`tree_param_paths`.
    matched : set[str]
        The names matching at least one pattern.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    resolved = tree_param_paths(tree)
    matched = {
        name for name in resolved if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    }
    return resolved, matched


def tree_with_fixed(tree, patterns: str | Sequence[str]):
    """
    Returns a tree with the parameters matching `patterns` frozen.

    Matches that are already frozen are left as they are, so one
    :func:`pmrf.unfreeze` undoes any number of calls.

    Parameters
    ----------
    tree : PyTree
        The tree to update.
    patterns : str or Sequence[str]
        `fnmatch` globs over parameter names, e.g. ``'load.*'``.

    Returns
    -------
    PyTree
        The updated tree.
    """
    from pmrf.utils.tree import freeze
    resolved, matched = _match_names(tree, patterns)
    matched = [name for name in matched if not _is_frozen_path(tree, resolved[name][0])]
    paths = [resolved[name][0] for name in matched]
    return _set_paths(tree, paths, [freeze(resolved[name][1]) for name in matched])


def tree_with_free(tree, patterns: str | Sequence[str]):
    """
    Returns a tree in which exactly the parameters matching `patterns` are free.

    All other parameters are frozen. Parameters fixed by construction (e.g.
    :func:`pmrf.Fixed`) stay fixed even if matched.

    Parameters
    ----------
    tree : PyTree
        The tree to update. May already be frozen, including nested freezes.
    patterns : str or Sequence[str]
        `fnmatch` globs over parameter names, e.g. ``'load.*'``.

    Returns
    -------
    PyTree
        The updated tree.
    """
    from pmrf.utils.tree import freeze, unfreeze
    tree = unfreeze(tree)
    resolved, matched = _match_names(tree, patterns)
    others = [name for name in resolved if name not in matched]
    return _set_paths(
        tree, [resolved[name][0] for name in others], [freeze(resolved[name][1]) for name in others]
    )


Space = Literal["physical", "unconstrained"]


def _is_free_node(x: Any) -> bool:
    """Partition filter for :func:`flatten`, applied at :func:`is_leaf` boundaries."""
    if is_param(x):
        return not x.fixed
    return prx.constraints.is_dynamic(x) and not isinstance(x, np.ndarray)


def _unwraps_whole(x: Any) -> bool:
    """Nodes :func:`flatten` unwraps in one step, so parameters carry their scale."""
    return is_param(x) or prx.is_constrained(x)


def _node_constraint_and_scale(node: Any) -> tuple[AbstractConstraint, float]:
    """Returns the unscaled constraint of a free node, and the scale taking it to physical."""
    if is_param(node):
        if prx.is_constrained(node.raw_value):
            return prx.as_unwrapped(node.raw_value.constraint), node.scale
        return prx.constraints.RealLine(shape=jnp.shape(node.unscaled_value)), node.scale
    return prx.constraints.tree_constraints(node), 1.0


def _element_names(base: str, shape: tuple[int, ...]) -> list[str]:
    """Expands a name over the elements of an array, in C order."""
    if shape == ():
        return [base]
    return [f"{base}[{','.join(map(str, index))}]" for index in np.ndindex(*shape)]


class FlatParams(eqx.Module):
    """
    A tree's free parameters as a flat, named vector.

    Created by :func:`flatten`. Every method is a pure JAX function of `theta`, so
    it can be passed through `jax.jit`, `jax.grad` and `jax.vmap`.

    **Density space.** Priors are authored on each parameter's unscaled value. The
    scale is folded into the density (see :func:`node_distribution`), so in
    ``'physical'`` space :meth:`log_prior` is a density over the scaled values in
    `theta`. In ``'unconstrained'`` space it is a density over `theta` itself: it
    adds the log-determinant of the Jacobian of the map to physical values,

    $$\\log p_\\theta(\\theta) = \\log p_x(f(\\theta)) + \\log\\left|\\det \\frac{\\partial f}{\\partial \\theta}\\right|,$$

    where $f$ is each parameter's constraint bijector followed by its scale.
    Parameters without a prior contribute nothing (an improper flat prior), and the
    priors of fixed or frozen parameters are included as constants, matching
    :class:`pmrf.problems.PriorPenalized`.
    """
    #: The name of each element of `theta`, in order.
    names: tuple[str, ...] = field(static=True)

    #: The tree's current free parameters as a flat vector.
    theta0: Array

    #: The space `theta` lives in.
    space: Space = field(static=True)

    #: The free nodes of the tree, still wrapped, with `None` elsewhere.
    _dynamic: Any
    #: The rest of the tree, with `None` in place of free nodes.
    _static: Any
    #: The unscaled constraint of each free node.
    _constraint: Any
    #: The scale of each free node.
    _scales: Any = field(static=True)
    #: Prior distributions mirroring the unwrapped tree.
    _distributions: Any
    #: Maps a flat physical vector back to the tree of free physical values.
    _unravel: Callable = field(static=True)

    def _scale(self, tree, inverse: bool = False):
        op = (lambda v, s: v / s) if inverse else (lambda v, s: v * s)
        return jax.tree.map(lambda s, sub: jax.tree.map(lambda v: op(v, s), sub), self._scales, tree)

    def _log_abs_scale(self, tree):
        per_node = jax.tree.map(
            lambda s, sub: sum(jnp.size(v) for v in jax.tree.leaves(sub)) * jnp.log(jnp.abs(s)),
            self._scales, tree,
        )
        return sum(jax.tree.leaves(per_node), start=jnp.asarray(0.0))

    def _physical_tree(self, tree):
        """Returns the free physical values of a tree with the same structure."""
        dynamic, _ = eqx.partition(tree, _is_free_node, is_leaf=is_leaf)
        return prx.unwrap(dynamic, only_if=_unwraps_whole)

    def _ravel(self, tree):
        """Returns `theta` for a tree with the same structure."""
        return self._theta_from_physical(self._physical_tree(tree))

    def _theta_from_physical(self, physical):
        """Returns `theta` for a tree of free physical values."""
        if self.space == "unconstrained" and self._constraint is not None:
            physical = self._constraint.bijector.inverse(self._scale(physical, inverse=True))
        return jax.flatten_util.ravel_pytree(physical)[0]

    def _to_physical(self, theta):
        """Returns the tree of free physical values at `theta`, and the log|det J| to it."""
        tree = self._unravel(theta)
        if self.space == "physical" or self._constraint is None:
            return tree, jnp.asarray(0.0)
        is_constraint = prx.constraints.is_constraint
        values = jax.tree.map(
            lambda c, sub: c.bijector.forward(sub), self._constraint.tree, tree, is_leaf=is_constraint
        )
        log_dets = jax.tree.map(
            lambda c, sub: jnp.sum(c.bijector.forward_log_det_jacobian(sub)),
            self._constraint.tree, tree, is_leaf=is_constraint,
        )
        log_det = sum(jax.tree.leaves(log_dets), start=jnp.asarray(0.0))
        return self._scale(values), log_det + self._log_abs_scale(values)

    def _unflatten_tree(self, physical):
        """Returns the unwrapped tree for a tree of free physical values."""
        return prx.unwrap(eqx.combine(self._static, physical, is_leaf=is_leaf))

    def _wrap_tree(self, physical):
        """Returns the wrapped free nodes for a tree of free physical values."""
        def wrap_node(node, value):
            if is_param(node):
                return node.wrap(value)
            if prx.is_wrappable(node):
                return prx.wrap(node, value)
            return value
        return jax.tree.map(wrap_node, self._dynamic, physical, is_leaf=is_leaf)

    def unflatten(self, theta: ArrayLike):
        """
        Returns the unwrapped tree at `theta`, ready for evaluation (e.g. ``.s(freq)``).

        Parameters
        ----------
        theta : ArrayLike
            A 1-D array in :attr:`space`, aligned with :attr:`names`.
        """
        return self._unflatten_tree(self._to_physical(jnp.asarray(theta))[0])

    def wrap(self, theta: ArrayLike):
        """
        Returns the wrapped tree at `theta`, keeping priors, constraints and fixed state.

        Bounds are not checked. Use this to save a result or to flatten again.

        Parameters
        ----------
        theta : ArrayLike
            A 1-D array in :attr:`space`, aligned with :attr:`names`.
        """
        physical = self._to_physical(jnp.asarray(theta))[0]
        return eqx.combine(self._static, self._wrap_tree(physical), is_leaf=is_leaf)

    def log_prior(self, theta: ArrayLike) -> Array:
        """
        Returns the log prior density at `theta`, in :attr:`space`.

        Parameters
        ----------
        theta : ArrayLike
            A 1-D array in :attr:`space`, aligned with :attr:`names`.

        Returns
        -------
        jax.Array
            A scalar. In ``'unconstrained'`` space it includes the log-determinant of
            the Jacobian to physical space.
        """
        physical, log_det = self._to_physical(jnp.asarray(theta))
        return tree_param_log_prob(self._distributions, self._unflatten_tree(physical)) + log_det


def flatten(tree, space: Space = "physical") -> FlatParams:
    """
    Flattens a tree's free parameters into a named 1-D vector.

    For external samplers and pipelines, which need a flat vector, its names, a map
    back to the model and a log prior. :func:`pmrf.optimize.run_minimizer` and
    :func:`pmrf.infer.run_sampler` are built on it.

    Names are those of :meth:`pmrf.Module.named_params` with ``free_only=True``, in
    the same order. An array-valued parameter expands into one name per element, in
    C order: ``coeffs[0]``, ``coeffs[1]``, and ``w[1,2]`` for an N-D array. A free
    node the name resolver does not see into, such as a joint prior's target, is
    named from its path.

    Parameters
    ----------
    tree : PyTree
        The tree to flatten, still wrapped.
    space : {'physical', 'unconstrained'}, default='physical'
        ``'physical'`` holds scaled parameter values, as in :meth:`pmrf.Module.values`.
        ``'unconstrained'`` maps each through its constraint bijector onto the real line.

    Returns
    -------
    FlatParams
        The flat vector with :meth:`~FlatParams.unflatten`, :meth:`~FlatParams.wrap`
        and :meth:`~FlatParams.log_prior`.

    Raises
    ------
    ValueError
        If `space` is not recognised.
    """
    if space not in ("physical", "unconstrained"):
        raise ValueError(f"`space` must be 'physical' or 'unconstrained', not {space!r}.")

    dynamic, static = eqx.partition(tree, _is_free_node, is_leaf=is_leaf)
    nodes = jax.tree_util.tree_flatten_with_path(dynamic, is_leaf=is_leaf)[0]
    constraint_tree = jax.tree.map(lambda n: _node_constraint_and_scale(n)[0], dynamic, is_leaf=is_leaf)
    scales = jax.tree.map(lambda n: _node_constraint_and_scale(n)[1], dynamic, is_leaf=is_leaf)

    physical = prx.unwrap(dynamic, only_if=_unwraps_whole)
    _, unravel = jax.flatten_util.ravel_pytree(physical)

    by_path = {path: name for name, (path, _) in tree_param_paths(tree).items()}
    names = []
    for path, node in nodes:
        base = by_path.get(path) or path_to_name(
            tree, path, namespace_separator='_', is_transparent=_is_name_transparent
        )
        value = prx.unwrap(node) if _unwraps_whole(node) else node
        for subpath, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
            names.extend(_element_names(base + jax.tree_util.keystr(subpath), jnp.shape(leaf)))

    flat = FlatParams(
        names=tuple(names),
        theta0=jnp.zeros(len(names)),
        space=space,
        _dynamic=dynamic,
        _static=static,
        _constraint=prx.constraints.Leafwise(constraint_tree) if nodes else None,
        _scales=scales,
        _distributions=tree_param_distributions(tree),
        _unravel=unravel,
    )
    return eqx.tree_at(lambda f: f.theta0, flat, flat._ravel(tree))


__all__ = [
    "flatten",
    "FlatParams",
    "Param",
    "is_param",
    "as_param",
    "param",
    "Fixed",
    "Unconstrained",
    "Bounded",
    "Constrained",
    "Random",
]
