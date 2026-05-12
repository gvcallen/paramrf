import jax
from typing import TypeVar, Callable, Any
import equinox as eqx

from equinox import field as field, Partial as Partial
from parax import unwrap as unwrap, as_free as as_free, as_fixed as as_fixed, as_frozen as as_frozen

T = TypeVar('T')

def combine(
    *trees: Any, 
    is_priority: Callable[[Any], bool] = None, 
    is_leaf: Callable[[Any], bool] = None
) -> Any:
    """
    Wrapper on top of `equinox.combine` that provides an optional
    `is_priority` argument.
    
    Combines trees by picking the first leaf that satisfies `is_priority`.
    If no leaves satisfy the condition, picks the first non-None leaf.
    """
    if is_priority is None:
        return eqx.combine(*trees, is_leaf=is_leaf)
    
    def _merge_leaves(*leaves):
        candidates = [l for l in leaves if l is not None]
        if not candidates:
            return None
        
        for leaf in candidates:
            if is_priority(leaf):
                return leaf
        
        return candidates[0]

    return jax.tree.map(_merge_leaves, *trees, is_leaf=is_leaf)