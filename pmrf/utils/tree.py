from typing import Any, Callable

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
    is_param as is_param,
)


from dataclasses import (
    InitVar as InitVar,
    replace as replace,
)


def freeze(value: Any):
    """
    Freezes/fixes a parameter or entire model and returns the frozen model.

    This can be used to freeze models to make them non-optimizable,
    but should also be used as a field converter (using `prf.field(converter=prf.fix)`)
    when storing raw arrays within in a model.
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


def is_model(x: Any):
    """
    Returns if `x` is an instance of :class:`pmrf.Model`.
    """
    from pmrf.models import Model
    return isinstance(x, Model)


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
