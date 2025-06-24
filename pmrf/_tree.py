from functools import reduce
from typing import Any, Callable, Dict
import re

import pmrf.numpy as np
from pmrf._misc import update_dict_with_alias
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

def path_to_param_name(
    path,
    leaf,
    subtree_separator: str,
    array_separator: str,
    index_separator: str,
) -> str | list[str]:
    path_str = subtree_separator.join(key.name for key in path if isinstance(key, GetAttrKey))

    param_names = None
    if np.isscalar(leaf):
        param_names = path_str
    else:
        # For array parameters, create an entry for each scalar element
        param_names = []
        for index in ndindex(leaf.shape):
            index_str = index_separator.join(map(str, index))
            param_names.append(f"{path_str}{array_separator}{index_str}")

    return param_names

def params_array(
    tree: Any,
) -> jax.Array:
    flat_leaves, _ = jax.tree.flatten(tree)
    if not flat_leaves:
        return np.array([]) # Return empty array if no params
    
    return np.concatenate([p.ravel() for p in flat_leaves])

def params_dict(
    tree: Any,
    separator: str | None = '_',
    subtree_separator: str | None = None,
    array_separator: str | None = None,
    index_separator: str | None = None,
    param_aliases: dict[str, str] | list[str] | None = None,
) -> Dict[str, Any]:
    subtree_separator: str = subtree_separator if subtree_separator is not None else separator
    array_separator: str = array_separator if array_separator is not None else separator
    index_separator: str = index_separator if index_separator is not None else separator
    
    paths_and_leaves = jax.tree.leaves_with_path(tree)

    parameters = {}
    for path, leaf in paths_and_leaves:
        param_name = path_to_param_name(path, leaf, subtree_separator, array_separator, index_separator)

        if np.isscalar(leaf):
            parameters[param_name] = leaf
        else:
            for name, index in zip(param_name, ndindex(leaf.shape)):
                parameters[name] = leaf[index]

    if not param_aliases is None:
        if isinstance(param_aliases, list):
            parameters = dict(zip(param_aliases, parameters.values()))
        else:
            raise Exception("Dict parameter aliases not yet supported")

    return parameters

def param_names_tree(
    tree: Any,
    separator: str | None = '_',
    subtree_separator: str | None = None,
    array_separator: str | None = None,
    index_separator: str | None = None,
) -> Any:
    subtree_separator: str = subtree_separator if subtree_separator is not None else separator
    array_separator: str = array_separator if array_separator is not None else separator
    index_separator: str = index_separator if index_separator is not None else separator
    
    paths_and_leaves, treedef = jax.tree.flatten_with_path(tree)
    param_names = []
    for path, leaf in paths_and_leaves:
        param_name = path_to_param_name(path, leaf, subtree_separator, array_separator, index_separator)
        param_names.append(param_name)

    return jax.tree.unflatten(treedef, param_names)
    
def with_params_from_array(
    tree: Any,
    params: jax.Array | None = None,
) -> Any:
    flat_leaves, treedef = jax.tree.flatten(tree)
    num_expected_params = sum(p.size for p in flat_leaves)
    
    # Ensure input is a JAX array for consistency
    params = np.asarray(params)

    if params.size != num_expected_params:
        raise ValueError(f"Input `flat_params` has size {params.size}, "
                            f"but model requires {num_expected_params}.")

    # Unflatten the leaves into a PyTree with the original structure.
    leaves = []
    offset = 0
    for leaf in flat_leaves:
        end = offset + leaf.size
        leaves.append(params[offset:end].reshape(leaf.shape))
        offset = end

    new_tree = jax.tree.unflatten(treedef, leaves)
    return new_tree

def with_params_from_dict(
    tree: Any,
    separator: str | None = '_',
    subtree_separator: str | None = None,
    array_separator: str | None = None,
    index_separator: str | None = None,
    param_aliases: dict[str, str] | list[str] | None = None,
    **params: Any
) -> Any:
    prev_params = params_dict(tree, separator=separator, subtree_separator=subtree_separator, array_separator=array_separator, index_separator=index_separator)
    
    if not param_aliases is None:
        if isinstance(param_aliases, list):
            param_aliases = {original: alias for original, alias in zip(prev_params.keys(), param_aliases)}
        update_dict_with_alias(prev_params, params, param_aliases)
    else:
        prev_params.update(params)
    return with_params_from_array(tree, list(prev_params.values()))

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
    core_spec: PyTree[AxisSpec],
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[PyTree, PyTree]:
    core, alias = eqx.partition(tree, core_spec)
    
    base_ids = jax.tree.map(lambda node: id(node), core, is_leaf=is_leaf)
    paths, ids = zip(*jax.tree.leaves_with_path(base_ids))
    id_to_path = dict(zip(ids, paths))
    ref = jax.tree.map(lambda node: RefNode(id_to_path[id(node)]), alias, is_leaf=is_leaf)
    
    return core, ref
    
def restore(
    ref: PyTree,
    core: PyTree | None = None,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    core = core if not core is None else ref
    def _is_leaf(node):
        is_leaf_val = False if is_leaf is None else is_leaf(node)
        return is_leaf_val or isinstance(node, RefNode)

    deref = jax.tree.map(lambda node: value_at_path(core, node.path) if isinstance(node, RefNode) else node, ref, is_leaf=_is_leaf)
    return deref

def partition(
    pytree: PyTree,
    filter_spec: PyTree[AxisSpec],
    shared_spec: PyTree[AxisSpec] | None = None,
    replace: Any = None,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[PyTree, PyTree]:
    if shared_spec is None:
        return eqx.partition(pytree, filter_spec, replace=replace, is_leaf=is_leaf)

    first, second = eqx.partition(pytree, shared_spec, replace=replace, is_leaf=is_leaf)
    first_core, first_ref = dealias(first, filter_spec, is_leaf)
    return first_core, eqx.combine(first_ref, second)

def combine(*pytrees: PyTree, restore = True, is_leaf: Callable[[Any], bool] | None = None) -> PyTree:
    combined = eqx.combine(*pytrees)
    from pmrf import _tree
    if restore:
        combined = _tree.restore(combined, is_leaf=is_leaf)
    return combined
