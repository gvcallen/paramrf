from typing import Any, Callable

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


def tie(
    tree: Any, 
    target: Callable[[Any], Any], 
    source: Callable[[Any], Any], 
    tie_fn: Callable[[Any], Any] = lambda x: x
):
    """(experimental) Allows tieing different nodes in a PyTree together.

    To evaluate the tie, use :func:`prf.unwrap`.

    Args:
        tree: The root PyTree or Equinox module to wrap.
        target: A callable (lens) that extracts the parameter to be replaced 
            (e.g., `lambda m: m.layer.weight`).
        source: A callable (lens) that extracts the parameter to draw values 
            from (e.g., `lambda m: m.layer.bias`).
        tie_fn: An optional transformation function applied to the source 
            parameter before injecting it into the target. Defaults to the 
            identity function.
    """
    return prx.Tie(tree, target=target, source=source, tie_fn=tie_fn)


def freeze(model: Any):
    """
    Freezes a model (or any JAX PyTree) and returns the frozen model.

    This can be used to freeze models to make them non-optimizable,
    but should also be used as a field converter (using `prf.field(converter=prf.freeze)`)
    when storing raw arrays within in a model.
    """
    return prx.as_opaque(model)


def unfreeze(model: Any):
    """
    Unfreezes a potentially frozen model and returns the unfrozen model.
    """
    model = prx.as_free(model)
    if isinstance(model, prx.Static):
        model = model.unwrap()
    return model


def is_model(x: Any):
    """
    Returns if `x` is an instance of :class:`pmrf.Model`.
    """
    from pmrf.models import Model
    return isinstance(x, Model)


def batched_tree_flatten(batched_tree: PyTree) -> Float[Array, "N D"]:
    """
    Flattens a batched PyTree into a 2D matrix of shape (N, D).
    
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
    template_sample: PyTree
) -> PyTree:
    """
    Restores a 2D matrix into a batched PyTree based on a template.

    Parameters
    ----------
    flat_array : Array
        The (N, D) array to unflatten.
    template_sample : PyTree
        A single-sample PyTree (e.g., your `y_init` or `f_struct`) 
        used to define the structure and leaf shapes.

    Returns
    -------
    PyTree
        A PyTree with the same structure as `template_sample`, where each 
        leaf has a leading dimension of size N.
    """
    # 1. Get the structure and sizes of a single sample
    leaves, treedef = jax.tree_util.tree_flatten(template_sample)
    leaf_sizes = [jnp.size(l) for l in leaves]
    leaf_shapes = [l.shape for l in leaves]
    
    # 2. Calculate split indices for the feature dimension
    # We drop the last index because jnp.split expects N-1 split points
    indices = jnp.cumsum(jnp.array(leaf_sizes))[:-1]
    
    # 3. Split the flat array along the feature axis (axis 1)
    batch_size = flat_array.shape[0]
    split_arrays = jnp.split(flat_array, indices, axis=-1)
    
    # 4. Reshape each chunk back to (N, *original_leaf_shape)
    restored_leaves = [
        arr.reshape(batch_size, *shape) 
        for arr, shape in zip(split_arrays, leaf_shapes)
    ]
    
    # 5. Reconstruct the PyTree
    return jax.tree_util.tree_unflatten(treedef, restored_leaves)
