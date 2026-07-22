"""
Parameter factories, converters, and field specifiers.

Most of these are re-exported at root.

Builds on top of `Parax <https://gvcallen.github.io/parax>`_.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Self, Union, Callable, TypeVar, TypeGuard

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
from pmrf.utils.optix import focus, Lens
from pmrf.utils.tree import filtered_pathed_leaves, path_to_name


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
            The raw Parax variable to wrap. Mutually exclusive with `value`.
        """
        if isinstance(value, prx.AbstractVariable):
            raise ValueError("Got a Parax variable when constructing a parameter")
        
        if raw_value is not None and value is not None:
            raise ValueError("Cannot pass `raw_value` and `value` to Param constructor")

        if raw_value is None:
            distribution, constraint = prx.unwrap(distribution), prx.unwrap(constraint)
            
            # Error Checking & Value Inference
            if value is None and distribution is None and constraint is None:
                raise ValueError("`value` was None when constructing a parameter but neither a distribution nor a finite Interval constraint was provided")
            if constraint is not None and value is not None:
                value_array = jnp.array(value)
                error_if(
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
    under :meth:`pmrf.Model.named_params`. It can also be used to enforce
    constraints, scaling, bounds and variability within the model itself.
    
    This simply creates a `pmrf.field` with a `pmrf.as_param` converter.
    
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
    over a sub-tree, such as by :class:`pmrf.models.Probabilistic`.

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


def tree_pathed_params(
    tree,
    full_params: bool = False,
    free_only: bool = False,
    keystr: bool = False,
    separator: str | None = None,
) -> list[tuple[Any, float | jnp.ndarray | Param]]:
    """
    Returns the parameters as a list of tuples alongside their paths.
    
    The paths represent JAX tree paths.
    
    Parameters
    ----------
    full_params : bool, default=True
        Returns the full parameter objects as opposed to their resultant floats/array values.
    free_only : bool, default=False
        Returns only free parameters.
    keystr : bool, default=False
        Whether equivalent strings should be returned as opposed to full JAX paths.
        Defaults to False.
    separator : str, optional
        The separator to use if `keystr` is True.
    """
    # Setup callables for filtering/flattening.
    # We also need to stop at parax's opaque/protected boundaries (e.g. `parax.Probabilize`),
    # via `prx.constraints.is_leaf` (which also covers frozen sub-trees via `prx.is_constant`).
    # Otherwise this recurses into e.g. a normalizing-flow prior's internals (its own trainable
    # weights, and the frozen reconstruction data), surfacing them as spurious named parameters.
    if free_only:
        filter_spec = lambda x: is_param(x) and not x.fixed or isinstance(x, jax.Array)
    else:
        filter_spec = lambda x: is_param(x) or isinstance(x, jax.Array)

    return filtered_pathed_leaves(tree, filter_spec, is_leaf=is_leaf, unwrap_leaves=not full_params, keystr=keystr, separator=separator)


def tree_named_params(
    tree,
    full_params: bool = False,
    free_only: bool = False,
    namespace_separator: str = '_',
) -> dict[str, float | jnp.ndarray | Param]:
    """
    Returns a named dictionary of parameters in a tree.
        
    Parameters
    ----------
    full_params : bool, default=True
        Returns the full parameter objects as opposed to their resultant floats/array values.
    free_only : bool, default=False
        Returns only free parameters.
    namespace_separator : str
        The separator to use to create a parameter namespace using model names.
    
    Returns
    -------
    dict[str, Any]
        A dictionary mapping string paths (e.g., '.ind.value') to their 
        corresponding JAX arrays or parameter objects.
        
    """
    pathed = tree_pathed_params(tree, free_only=free_only, full_params=full_params)
    
    # Detect collisions
    named = {}
    for path, leaf in pathed:
        name = path_to_name(tree, path, namespace_separator=namespace_separator)
        if name in named:
            raise ValueError(
                f"Parameter name collision: '{name}'.\n\n"
                f"Multiple paths resolved to the same name during flattening. "
                f"To fix this, either assign unique names directly to the parameters, "
                f"or give their parent models distinct names to create unique prefixes."
            )
        named[name] = leaf
    
    return named

def tree_param_names_to_path(tree):
    pathed = tree_pathed_params(tree, full_params=True)
    name_to_path = {}
    for path, _ in pathed:
        name = path_to_name(tree, path, namespace_separator='_')
        name_to_path[name] = path    

    return name_to_path


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
]