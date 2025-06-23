from functools import reduce
from typing import Any, Callable, Dict
import re

import pmrf.numpy as np
from numpy import ndindex
import jax
from jax.tree_util import SequenceKey, DictKey, GetAttrKey
import equinox as eqx

from typing import Any, Tuple, List, Type
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, fields

import jax
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree, PyTreeDef
from jax.tree_util import DictKey, SequenceKey, GetAttrKey

AxisSpec = bool | Callable[[Any], bool]

def with_params_from_dict(
    tree: Any,
    separator: str | None = '_',
    subtree_separator: str | None = None,
    array_separator: str | None = None,
    index_separator: str | None = None,
    param_filter: PyTree | Callable[[Any], bool] | None = None,
    **params: Any
) -> Any:
    """
    Returns a new tree with updated parameter values by key-word.

    Args:
        tree: The original, immutable PyTree (e.g., an Equinox model).
        separator: The default separator used if others are not specified.
        subtree_separator: Separator for nested attribute names.
        array_separator: Separator between attribute path and array index.
        index_separator: Separator for multi-dimensional array indices.
        param_filter: A function that returns `True` for nodes in the tree that are parameters.
        **params: Keyword arguments where keys are the string paths of the
                  parameters to update and values are their new values.
    """
    # --- Initial argument validation and setup ---
    param_filter = param_filter or eqx.is_array
    
    # Set default separators for consistency with tree_parameter_paths
    subtree_separator = subtree_separator if subtree_separator is not None else separator
    array_separator = array_separator if array_separator is not None else separator
    index_separator = index_separator if index_separator is not None else separator
    
    # We will apply updates sequentially. Start with the original tree.
    new_tree = tree

    for path_str, value in params.items():
        # Regex to separate the base path from the array index, e.g., "sub_C" from "[0_0]"
        # It looks for the array_separator followed by anything.
        match = re.match(rf"^(.*?){re.escape(array_separator)}(.*)$", path_str)

        if match:
            base_path_str, index_str = match.groups()
            # Parse index string "0_0" into a tuple of ints (0, 0)
            index = tuple(map(int, index_str.split(index_separator)))
        else:
            base_path_str = path_str
            index = None # This is a scalar or full-array update

        # Parse base path "sub_R" into a list of attribute names ['sub', 'R']
        attr_names = base_path_str.split(subtree_separator)

        # Create the `where` function to navigate to the target attribute
        # `reduce` applies `getattr` sequentially to walk the path
        where = lambda m: reduce(getattr, attr_names, m)

        # If there's an index, the `where` function must also apply it
        if index is not None:
            # We need a new lambda to capture the current `where` and `index`
            # This is safer than modifying the lambda in-place
            base_where = where
            where = lambda m: base_where(m)[index]

        # Apply the single update and reassign the tree for the next loop iteration
        new_tree = eqx.tree_at(where, new_tree, value)

    return new_tree
    
def with_params_from_array(
    tree: Any,
    params: jax.Array | None = None,
    param_filter: PyTree | Callable[[Any], bool] | None = None,
) -> Any:
    """
    Returns a new tree with updated parameter values.

    Args:
        tree: The original, immutable PyTree (e.g., an Equinox model).
        params: A 1D JAX array containing all dynamic parameter
                     values in their flattened tree order.
        param_filter: A function that returns `True` for nodes in the tree that are parameters.
    """
    # --- Initial argument validation and setup ---
    param_filter = param_filter or eqx.is_array

    params_tree, static = partition(tree, param_filter)
    flat_leaves, treedef = jax.tree.flatten(params_tree)
    num_expected_params = sum(p.size for p in flat_leaves)
    
    # Ensure input is a JAX array for consistency
    params = np.asarray(params)

    if params.size != num_expected_params:
        raise ValueError(f"Input `flat_params` has size {params.size}, "
                            f"but model requires {num_expected_params}.")

    # Unflatten the leaves into a PyTree with the original structure.
    # Note: JAX can unflatten a 1D array into leaves of various shapes.
    leaves = []
    offset = 0
    for leaf in flat_leaves:
        end = offset + leaf.size
        leaves.append(params[offset:end].reshape(leaf.shape))
        offset = end

    new_params_tree = jax.tree.unflatten(treedef, leaves)
    return eqx.combine(new_params_tree, static)
    
    
def params_dict(
    tree: Any,
    separator: str | None = '_',
    subtree_separator: str | None = None,
    array_separator: str | None = None,
    index_separator: str | None = None,
    param_filter: PyTree | Callable[[Any], bool] | None = None,
) -> Dict[str, Any]:
    """
    Returns the dynamic parameters of a PyTree as a dictionary of
    paths and values.

    The order of the returned paths and values precisely matches the order
    from `jax.tree_util.tree_flatten`.

    Args:
        tree: The PyTree (e.g., an Equinox model) to inspect.
        flat: If `True`, returns a single 1D JAX array of all scalar parameter
              values. If `False` (default), returns a dictionary mapping
              human-readable string paths to their scalar values.
        separator: The default separator used if others are not specified.
        subtree_separator: Separator for nested attribute names.
        array_separator: Separator between attribute path and array index.
        index_separator: Separator for multi-dimensional array indices.
        param_filter: A function that returns `True` for nodes in the tree that are parameters.

    Returns:
        A dictionary of parameter names/paths and values.
    """
    param_filter = param_filter or eqx.is_array
    params_tree, _ = partition(tree, param_filter)

    # --- Logic Branch 2: Return a dictionary of paths and values ---
    # Set default separators, only needed for the dictionary case
    subtree_separator: str = subtree_separator if subtree_separator is not None else separator
    array_separator: str = array_separator if array_separator is not None else separator
    index_separator: str = index_separator if index_separator is not None else separator
    
    paths_and_leaves = jax.tree.leaves_with_path(params_tree)

    parameters = {}
    for path, leaf in paths_and_leaves:
        # Convert path tuple to a human-readable string, e.g., "sub.R"
        path_str = subtree_separator.join(key.name for key in path if isinstance(key, GetAttrKey))

        if leaf.ndim == 0:
            parameters[path_str] = leaf
        else:
            # For array parameters, create an entry for each scalar element
            for index in ndindex(leaf.shape):
                index_str = index_separator.join(map(str, index))
                parameters[f"{path_str}{array_separator}{index_str}"] = leaf[index]
    
    return parameters

def params_array(
    tree: Any,
    param_filter: PyTree | Callable[[Any], bool] | None = None,
) -> jax.Array:
    """
    Returns the dynamic parameters of a PyTree a single flattened JAX array.

    The order of the returned paths and values precisely matches the order
    from `jax.tree_util.tree_flatten`.

    Args:
        tree: The PyTree (e.g., an Equinox model) to inspect.
        param_filter: A function that returns `True` for nodes in the tree that are parameters.
    Returns:
        A single 1D JAX array of all parameter values.
    """
    param_filter = param_filter or eqx.is_array
    params_tree, _ = partition(tree, param_filter)
    flat_leaves, _ = jax.tree.flatten(params_tree)
    if not flat_leaves:
        return np.array([]) # Return empty array if no params
    
    # Concatenate all leaves into a single 1D vector
    return np.concatenate([p.ravel() for p in flat_leaves])

def flatten_one_level_with_path(
    pytree: Any, is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> tuple[list[PyTree], PyTreeDef]:
    # See eqx.tree_flatten_one_level
    seen_pytree = False
    
    def is_leaf(node):
        nonlocal seen_pytree
        if node is pytree:
            if seen_pytree:
                try:
                    type_string = type(pytree).__name__
                except AttributeError:
                    type_string = "<unknown>"
                raise ValueError(
                    f"PyTree node of type `{type_string}` is immediately "
                    "self-referential; that is to say it appears within its own PyTree "
                    "structure as an immediate subnode. (For example "
                    "`x = []; x.append(x)`.) This is not allowed."
                )
            else:
                seen_pytree = True
            return False
        else:
            return True

    return jax.tree.flatten_with_path(pytree, is_leaf=is_leaf, is_leaf_takes_path=is_leaf_takes_path)

def flatten_one_level_with_metadata(
    pytree: Any, is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> tuple[list[PyTree], PyTreeDef]:
    path_vals, treedef = flatten_one_level_with_path(pytree, is_leaf=is_leaf, is_leaf_takes_path=is_leaf_takes_path)
    name_to_metadata = {}
    for field in fields(pytree):
        name_to_metadata[field.name] = field.metadata
    
    flattened_metadata = []
    for path, val in path_vals:
        name = path[0].name
        if not name in name_to_metadata:
            raise Exception(f"{name} attribute not in metadata")
        flattened_metadata.append((name_to_metadata[name], val))
        
    return flattened_metadata, treedef

# def metadata(
#     pytree: Any, is_leaf: Callable[..., bool] | None = None,
#     is_leaf_takes_path: bool = False,
# ) -> PyTree:
#     parent_path_to_name_to_metadata = {}
#     def populate_metadata(path, node):
#         if is_dataclass(node) and len(path) < 2:
#             return (node, {})
#         parent_path = path[0:-1]
#         name = path[-1]
#         if not parent_path in parent_path_to_name_to_metadata:
#             name_to_metadata = {}
#             parent = value_at_path(parent, parent_path)
#             for field in fields(parent):
#                 name_to_metadata[field.name] = field.matadata
#             parent_path_to_name_to_metadata[parent_path] = name_to_metadata
#         return parent_path_to_name_to_metadata[parent_path][name]
    
#     return jax.tree.map_with_path(populate_metadata, pytree, is_leaf=is_leaf, is_leaf_takes_path=is_leaf_takes_path)

# def flatten_with_metadata(
#     pytree: Any, is_leaf: Callable[..., bool] | None = None,
#     is_leaf_takes_path: bool = False,
# ) -> tuple[list[PyTree], PyTreeDef]:
#     path_vals, treedef = jax.tree.flatten_with_path(pytree, is_leaf=is_leaf, is_leaf_takes_path=is_leaf_takes_path)
    
#     flattened_metadata = []
#     parent_path_to_name_to_metadata = {}
#     for path, _ in path_vals:
#         parent_path = path[0:-1]
#         if not parent_path in parent_path_to_name_to_metadata:                    
#             parent = value_at_path(pytree, parent_path)
#             name_to_metadata = {}
#             for field in fields(parent):
#                 name_to_metadata[field.name] = field.matadata
#             parent_path_to_name_to_metadata[parent_path] = name_to_metadata
#         metadata = parent_path_to_name_to_metadata[parent_path][path[-1].name]
#         name = path[-1].name
    
#     name_to_metadata = {}
#     for field in fields(pytree):
#         name_to_metadata[field.name] = field.metadata
    
#     flattened_metadata = []
#     for path, val in path_vals:
#         name = path[0].name
#         if not name in name_to_metadata:
#             raise Exception(f"{name} attribute not in metadata")
#         flattened_metadata.append((name_to_metadata[name], val))
        
#     return flattened_metadata, treedef

def nodes_by_type(tree: Any, match_type: Type) -> List[Tuple[Tuple[Any, ...], Any]]:
    matches = []

    if isinstance(tree, match_type):
        matches.append(tree)

    # Handle dataclasses
    if is_dataclass(tree) and not isinstance(tree, type):
        for f in fields(tree):
            value = getattr(tree, f.name)
            matches.extend(nodes_by_type(value, match_type))

    # Handle dicts
    elif isinstance(tree, Mapping):
        for k, v in tree.items():
            matches.extend(nodes_by_type(v, match_type))

    # Handle lists, tuples, etc.
    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        for i, v in enumerate(tree):
            matches.extend(nodes_by_type(v, match_type))

    return matches

def nodes_by_type_with_path(tree: Any, match_type: Type, path=()) -> List[Tuple[Tuple[Any, ...], Any]]:
    # TODO upgrade to ENSURE our paths are 100% jax compatible
    matches = []

    if isinstance(tree, match_type):
        matches.append((path, tree))

    # Handle dataclasses
    if is_dataclass(tree) and not isinstance(tree, type):
        for f in fields(tree):
            value = getattr(tree, f.name)
            matches.extend(nodes_by_type_with_path(value, match_type, path + (GetAttrKey(f.name),)))

    elif isinstance(tree, Mapping):
        for k, v in tree.items():
            matches.extend(nodes_by_type_with_path(v, match_type, path + (DictKey(k),)))

    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        for i, v in enumerate(tree):
            matches.extend(nodes_by_type_with_path(v, match_type, path + (SequenceKey(i),)))

    return matches

def value_at_path(pytree, path):
    node = pytree
    for key in path:
        if isinstance(key, GetAttrKey):
            k = key.name
            node = getattr(node, k)
        elif isinstance(key, SequenceKey):
            i = key.idx
            node = node[i]
        elif isinstance(key, DictKey):
            k = key.name
            node = node[key]
        else:
            raise Exception(f"Only DictKey, SequenceKey and GetAttrKey are supported in <node_at_path> but '{type(key)}' was passed of value {key}")
        
    return node

def values_at_paths(pytree, paths):
    nodes = []
    for path in paths:
        nodes.append(value_at_path(pytree, path))
    return nodes

def path_repr(path):
    repr = ""
    for key in path:
        if isinstance(key, GetAttrKey) or isinstance(key, DictKey):
            repr += f"['{key.name}']"
        elif isinstance(key, SequenceKey):
            repr += f"[{key.idx}]"
        else:
            raise Exception(f"Only DictKey, SequenceKey and GetAttrKey are supported in <path_repr> but '{type(key)}' was passed of value {key}")        
        
    return repr

class RefNode:
    def __init__(self, path):
        self.path = path
    def __repr__(self):
        return f"RefNode({self.path})"        
        
def dealias(
    tree: PyTree,
    base_spec: PyTree[AxisSpec],
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[PyTree, PyTree]:
    base, aliased = eqx.partition(tree, base_spec)
    
    base_ids = jax.tree.map(lambda node: id(node), base, is_leaf=is_leaf)
    paths, ids = zip(*jax.tree.leaves_with_path(base_ids))
    id_to_path = dict(zip(ids, paths))
    
    ref = jax.tree.map(lambda node: RefNode(id_to_path[id(node)]), aliased, is_leaf=is_leaf)
    return base, ref
    
def restore(
    core: PyTree,
    ref: PyTree,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    aliased = jax.tree.map(lambda ref_node: value_at_path(core, ref_node.path), ref, is_leaf=lambda node: is_leaf(node) or isinstance(node, RefNode))
    return core, aliased
