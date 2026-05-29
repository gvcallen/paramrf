from typing import Any, Callable, Generic, TypeVar, Any
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

import parax as prx

from equinox import (
    Partial as Partial,
    field as field,
    combine as combine,
)

from parax import (
    unwrap as unwrap,
    unwrap_self as unwrap_self,
    is_constant as is_constant,
)


from dataclasses import (
    InitVar as InitVar,
    replace as replace,
)


def freeze(value: Any):
    """
    Freezes/fixes a parameter or entire model and returns the frozen model.

    This can be used to freeze objects to make them non-optimizable.
    For example, it can be use as a field converter in custom models
    using `prf.field(converter=prf.freeze)` to store raw JAX arrays.
    """
    return prx.as_opaque(value)


def unfreeze(value: Any):
    """
    Unfreezes/unfixes a potentially frozen parameter or model and returns the unfrozen model.
    """
    value = prx.as_free(value)
    if isinstance(value, prx.Static):
        value = value.unwrap()
    return value


def extract_batch_axes(batched_tree: PyTree, template_tree: PyTree, *, is_leaf: Callable[[Any], bool] | None = None) -> PyTree:
    """
    (experimental) Generates an in_axes PyTree by comparing a batched model to a template.
    
    Parameters
    ----------
    batched_tree : PyTree
        A PyTree where every leaf has the same leading batch dimension.
    template_tree : PyTree
        A single-sample PyTree used to define the structure and leaf shapes.
    is_leaf : Callable[[Any], bool], optional
        An optional callable defining whether a node in the PyTree is a leaf.

    Returns
    -------
    Array
        The axis spec for the batched leafs, which can be used in functions like :func:`equinox.filter_vmap`.
    """
    
    def _compare(batched_leaf, template_leaf):
        # Check if the batched leaf has an 'ndim' attribute (is an array) 
        # and if its dimensions are greater than the template leaf (which could be a scalar)
        if hasattr(batched_leaf, "ndim") and batched_leaf.ndim > jnp.ndim(template_leaf):
            return 0
        return None
        
    return jax.tree.map(_compare, batched_tree, template_tree, is_leaf=is_leaf)

def batched_tree_flatten(batched_tree: PyTree) -> Float[Array, "N D"]:
    """
    (experimental) Flattens a batched PyTree into a 2D matrix of shape (N, D).
    
    Parameters
    ----------
    batched_tree : PyTree
        A PyTree where every leaf has a leading batch dimension of size N.

    Returns
    -------
    Array
        A single JAX array of shape (N, D), where D is the total number of 
        elements in a single sample of the PyTree.
    """
    leaves = jax.tree_util.tree_leaves(batched_tree)
    if not leaves:
        return jnp.zeros((0, 0))
    
    batch_size = leaves[0].shape[0]
    # Reshape each leaf to (N, -1) and concatenate along the feature axis
    flat_leaves = [leaf.reshape(batch_size, -1) for leaf in leaves]
    return jnp.concatenate(flat_leaves, axis=-1)


def batched_tree_unflatten(
    flat_array: Float[Array, "N D"], 
    template_tree: PyTree
) -> PyTree:
    """
    (experimental) Restores a 2D matrix into a batched PyTree based on a template.

    Parameters
    ----------
    flat_array : Array
        The (N, D) array to unflatten.
    template_tree : PyTree
        A single-sample PyTree used to define the structure and leaf shapes.

    Returns
    -------
    PyTree
        A PyTree with the same structure as `template_sample`, where each 
        leaf has a leading dimension of size N.
    """
    leaves, treedef = jax.tree.flatten(template_tree)
    leaf_sizes = [jnp.size(l) for l in leaves]
    leaf_shapes = [l.shape for l in leaves]
    
    indices = jnp.cumsum(jnp.array(leaf_sizes))[:-1]
    
    batch_size = flat_array.shape[0]
    split_arrays = jnp.split(flat_array, indices, axis=-1)
    
    restored_leaves = [
        arr.reshape(batch_size, *shape) 
        for arr, shape in zip(split_arrays, leaf_shapes)
    ]
    
    return jax.tree.unflatten(treedef, restored_leaves)


def filtered_pathed_leaves(
    tree: Any,
    filter_spec,
    is_leaf: Callable = None,
    unwrap: bool = True,
    keystr: bool = False,
    separator: str | None = None,
) -> list[tuple[Any, Any]]:
    # Get rid of any non-param leaves
    filtered_tree = eqx.filter(tree, filter_spec, is_leaf=is_leaf)
    pathed, _ = jax.tree.flatten_with_path(filtered_tree, is_leaf=is_leaf)
    
    if unwrap or keystr:
        for i in range(len(pathed)):
            key, value = pathed[i]
            
            if keystr:
                kwargs = {'separator': separator} if separator is not None else {}
                key = jax.tree_util.keystr(key, **kwargs)
            
            if unwrap:
                value = prx.unwrap(value)
                if jnp.isscalar(value):
                    value = float(value)
            
            pathed[i] = (key, value)
            
    return pathed


_Return = TypeVar("_Return")

class Bind(eqx.Module, Generic[_Return]):
    """(experimental) Like `functools.partial`, but allows re-passing keyword arguments 
    to override the originally bound keyword arguments.
    """

    func: Callable[..., _Return]
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __init__(self, func: Callable[..., _Return], /, *args: Any, **kwargs: Any):
        self.func = func
        self.args = args
        self.keywords = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> _Return:
        # Merge dictionaries: Call-time kwargs overwrite init-time self.keywords
        # If you are on Python 3.8 or older, use: {**self.keywords, **kwargs}
        merged_kwargs = self.keywords | kwargs 
        
        return self.func(*self.args, *args, **merged_kwargs)    
    

class Attrgetter(eqx.Module):
    """(experimental) Like `operator.attrgetter`, but registered as a PyTree.
    Supports single attributes, multiple attributes, and dotted paths.
    """
    attrs: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, *attrs: str):
        if not attrs:
            raise TypeError("Attrgetter expected at least 1 argument, got 0")
        self.attrs = attrs

    def __call__(self, obj: Any) -> Any:
        getter = operator.attrgetter(*self.attrs)
        return getter(obj)
    

class Pathgetter(eqx.Module):
    """(experimental) Retrieves values from a PyTree given JAX tree paths.
    Supports single paths or multiple paths, symmetric to `Attrgetter`.
    """
    
    # We store a tuple of tuples
    paths: tuple[tuple[Any, ...], ...] = eqx.field(static=True)

    def __init__(self, *paths: tuple[Any, ...] | list[Any]):
        if not paths:
            raise TypeError("Pathgetter expected at least 1 argument, got 0")
        
        # Cast every path to a tuple to guarantee strict hashability for JAX
        self.paths = tuple(tuple(p) for p in paths)

    def __call__(self, tree: Any) -> Any:
        def _get_single_path(current_obj: Any, path: tuple[Any, ...]) -> Any:
            for p in path:
                if hasattr(p, "name"):     # Handles GetAttrKey
                    current_obj = getattr(current_obj, p.name)
                elif hasattr(p, "idx"):    # Handles SequenceKey
                    current_obj = current_obj[p.idx]
                elif hasattr(p, "key"):    # Handles DictKey
                    current_obj = current_obj[p.key]
                else:
                    # Fallback for raw strings/ints in custom paths
                    try:
                        current_obj = getattr(current_obj, p)
                    except AttributeError:
                        current_obj = current_obj[p]
            return current_obj

        # Return a single value if only one path was requested
        if len(self.paths) == 1:
            return _get_single_path(tree, self.paths[0])
        
        # Return a tuple of values if multiple paths were requested
        return tuple(_get_single_path(tree, path) for path in self.paths)