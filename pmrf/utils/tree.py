from typing import Any, Callable, Tuple, List, Type
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, fields

import jax
from jax.tree_util import SequenceKey, DictKey, GetAttrKey
from jaxtyping import PyTree, PyTreeDef
from jax.tree_util import DictKey, SequenceKey, GetAttrKey
import equinox as eqx

from pmrf.constants import TreeAxisSpec

# Dummy node used for array sharing in a model. See partition(..)
class RefNode:
    """
    A placeholder node used to represent a reference to another node in a PyTree.

    This is used during partitioning/de-aliasing to handle shared parameters
    (object identity) within a model. Instead of duplicating a shared array,
    one instance is kept and others are replaced by a `RefNode` pointing to
    the path of the original instance.

    Attributes
    ----------
    path : tuple
        The JAX key path (e.g., tuple of GetAttrKey, etc.) to the referenced node.
    """
    def __init__(self, path):
        self.path = path
    def __repr__(self):
        return f"RefNode({tuple(self.path)})"

# Dummy node used for array sharing in a model. See partition(..)
def flatten_one_level_with_path(
    pytree: Any, is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> tuple[list[PyTree], PyTreeDef]:
    """
    Flatten a PyTree one level deep, returning values paired with their paths.

    This is similar to `equinox.tree_flatten_one_level`, but utilizes JAX's
    path-tracking capabilities.

    Parameters
    ----------
    pytree : Any
        The PyTree to flatten.
    is_leaf : Callable, optional
        A function to determine if a node is a leaf.
    is_leaf_takes_path : bool, optional, default=False
        If True, `is_leaf` accepts the path as a second argument.

    Returns
    -------
    list
        A list of `(path, child)` tuples.
    PyTreeDef
        The tree definition.

    Raises
    ------
    ValueError
        If the PyTree is immediately self-referential.
    """
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
    """
    Flatten a Dataclass PyTree one level, associating field metadata with values.

    Parameters
    ----------
    pytree : Any
        The dataclass instance to flatten.
    is_leaf : Callable, optional
        Leaf predicate.
    is_leaf_takes_path : bool, optional, default=False
        If True, `is_leaf` accepts the path.

    Returns
    -------
    list
        A list of `(metadata, value)` tuples.
    PyTreeDef
        The tree definition.

    Raises
    ------
    Exception
        If a field name in the flattened path is not found in the dataclass fields.
    """
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

def nodes_by_type(tree: Any, match_type: Type) -> List[Tuple[Tuple[Any, ...], Any]]:
    """
    Recursively search a PyTree for nodes matching a specific type.

    Parameters
    ----------
    tree : Any
        The PyTree to search.
    match_type : Type
        The class type to match.

    Returns
    -------
    List[Any]
        A list of matching node instances.
    """
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
    """
    Recursively search a PyTree for nodes matching a specific type, including their paths.

    Parameters
    ----------
    tree : Any
        The PyTree to search.
    match_type : Type
        The class type to match.
    path : tuple, optional
        The current path (accumulator).

    Returns
    -------
    List[Tuple[tuple, Any]]
        A list of `(path, node)` tuples.
    """
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
    """
    Retrieve a node from a PyTree using a JAX key path.

    Parameters
    ----------
    pytree : Any
        The root PyTree.
    path : tuple
        A sequence of JAX keys (`GetAttrKey`, `SequenceKey`, or `DictKey`).

    Returns
    -------
    Any
        The value found at the specified path.

    Raises
    ------
    Exception
        If the path contains unsupported key types or invalid keys.
    """
    node = pytree
    for item in path:
        if isinstance(item, GetAttrKey):
            k = item.name
            node = getattr(node, k)
        elif isinstance(item, SequenceKey):
            i = item.idx
            node = node[i]
        elif isinstance(item, DictKey):
            k = item.key
            try:
                node = node[k]
            except:
                raise Exception(f'Key error for path {path} with key "{k}"')
        else:
            raise Exception(f"Only DictKey, SequenceKey and GetAttrKey are supported in <node_at_path> but '{type(item)}' was passed of value {item} at path {path}")
        
    return node

def values_at_paths(pytree, paths):
    """
    Retrieve multiple nodes from a PyTree given a list of paths.

    Parameters
    ----------
    pytree : Any
        The root PyTree.
    paths : list[tuple]
        A list of JAX key paths.

    Returns
    -------
    list
        A list of values corresponding to the paths.
    """
    nodes = []
    for path in paths:
        nodes.append(value_at_path(pytree, path))
    return nodes

def path_repr(path):
    """
    Convert a JAX key path into a readable string representation.

    Parameters
    ----------
    path : tuple
        The JAX key path.

    Returns
    -------
    str
        String representation (e.g., ``['layer'][0]['weight']``).
    """
    repr = ""
    for key in path:
        if isinstance(key, GetAttrKey) or isinstance(key, DictKey):
            repr += f"['{key.name}']"
        elif isinstance(key, SequenceKey):
            repr += f"[{key.idx}]"
        else:
            raise Exception(f"Only DictKey, SequenceKey and GetAttrKey are supported in <path_repr> but '{type(key)}' was passed of value {key}")        
        
    return repr

def dealias_shared(
    tree: PyTree,
    core_spec: PyTree[TreeAxisSpec],
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[PyTree, PyTree]:
    """
    Split a PyTree into a 'core' and a 'ref' tree, handling object aliasing.

    This function identifies nodes that exist in the `core` partition (defined by
    `core_spec`). If those same nodes (identified by `id()`) appear in the `alias`
    partition, they are replaced in the `alias` partition by a `RefNode` pointing
    to their location in the `core`.

    This allows maintaining shared parameters (pointers) when partitioning models.

    Parameters
    ----------
    tree : PyTree
        The input tree.
    core_spec : PyTree[TreeAxisSpec]
        Specification defining which nodes belong to the core.
    is_leaf : Callable, optional
        Leaf predicate.

    Returns
    -------
    core : PyTree
        The partitioned core tree.
    ref : PyTree
        The partitioned alias tree, with shared nodes replaced by `RefNode`.
    """
    core, alias = eqx.partition(tree, core_spec, is_leaf=is_leaf)
    
    base_ids = jax.tree.map(lambda node: id(node), core, is_leaf=is_leaf)
    paths, ids = zip(*jax.tree.leaves_with_path(base_ids))
    id_to_path = dict(zip(ids, paths))
    
    def node_to_path(node):
        node_id = id(node)
        
        # If we find no aliases in the core, we simply return the node.
        # This was previously an error, however the function spec
        # is actually that we only dealias a node if it is *in* the core spec at all
        # (i.e. it can have aliases in the non-core spec)
        if not node_id in id_to_path:
            return node
        return RefNode(id_to_path[node_id])
    ref = jax.tree.map(node_to_path, alias, is_leaf=is_leaf)
    
    return core, ref
    
def restore_shared(
    ref: PyTree,
    core: PyTree | None = None,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """
    Restore aliased references in a tree using values from a core tree.

    Replaces any `RefNode` found in `ref` with the actual value located at
    `RefNode.path` within `core`.

    Parameters
    ----------
    ref : PyTree
        The tree containing `RefNode`s.
    core : PyTree, optional
        The source tree containing the actual values. If None, `ref` is used as source.
    is_leaf : Callable, optional
        Leaf predicate.

    Returns
    -------
    PyTree
        The restored tree with aliases resolved.
    """
    core = core if not core is None else ref
    def _is_leaf(node):
        is_leaf_val = False if is_leaf is None else is_leaf(node)
        return is_leaf_val or isinstance(node, RefNode)

    deref = jax.tree.map(lambda node: value_at_path(core, node.path) if isinstance(node, RefNode) else node, ref, is_leaf=_is_leaf)
    return deref

def partition_shared(
    pytree: PyTree,
    filter_spec: PyTree[TreeAxisSpec],
    shared_spec: PyTree[TreeAxisSpec] | None = None,
    replace: Any = None,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[PyTree, PyTree]:
    """
    Partition a PyTree with support for shared references.

    Combines standard `equinox.partition` with `dealias` logic. If `shared_spec`
    is provided, it partitions the tree and ensures that shared nodes are
    represented by references rather than duplicates.

    Parameters
    ----------
    pytree : PyTree
        The tree to partition.
    filter_spec : PyTree[TreeAxisSpec]
        The specification for the primary partition.
    shared_spec : PyTree[TreeAxisSpec], optional
        The specification used for de-aliasing. If None, behaves like standard partition.
    replace : Any, optional
        Value to replace filtered nodes with (default None).
    is_leaf : Callable, optional
        Leaf predicate.

    Returns
    -------
    tuple[PyTree, PyTree]
        The partitioned (and potentially de-aliased) trees.
    """
    if shared_spec is None or filter_spec == shared_spec:
        return eqx.partition(pytree, filter_spec, replace=replace, is_leaf=is_leaf)

    first, second = eqx.partition(pytree, shared_spec, replace=replace, is_leaf=is_leaf)
    first_core, first_ref = dealias_shared(first, filter_spec, is_leaf)
    
    return first_core, eqx.combine(first_ref, second)

def combine_shared(*pytrees: PyTree, restore = True, is_leaf: Callable[[Any], bool] | None = None) -> PyTree:
    """
    Combine multiple PyTrees, optionally restoring shared references.

    Parameters
    ----------
    *pytrees : PyTree
        The trees to combine.
    restore : bool, optional, default=True
        If True, resolves `RefNode` aliases after combination.
    is_leaf : Callable, optional
        Leaf predicate.

    Returns
    -------
    PyTree
        The combined tree.
    """
    combined = eqx.combine(*pytrees, is_leaf=is_leaf)
    from pmrf.utils import tree
    if restore:
        combined = tree.restore_shared(combined, is_leaf=is_leaf)
    return combined