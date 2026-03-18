from typing import TypeVar

import jax
import equinox as eqx
from pmrf.utils import is_valid_param

T = TypeVar("T")

def partition(pytree: T, include_fixed=False, param_objects=False) -> tuple[T, T]:
    """
    Partitions an arbitrary PyTree into (dynamic, static) halves.

    This upgrades `eqx.partition` to be "parameter-aware",
    treating only :class:`pmrf.core.parameter` objects as dynamic and
    also taking into account whether they are fixed or not.
    
    Args:
        pytree: The Problem, Model, or any PyTree containing Parameters.
        include_fixed: If True, includes parameters where fixed=True.
        param_objects: If True, the entire Parameter object is routed to the dynamic tree. 
                       If False (default), ONLY the .value array is routed to the dynamic tree.
                       
    Returns:
        tuple: (dynamic, static) PyTrees suitable for Optimistix or Numpyro.
    """
    
    def build_mask(node):
        # 1. Safely check if the current node is a Parameter
        if is_valid_param(node):
            
            # 2. Handle fixed parameters cleanly
            if not include_fixed and getattr(node, "fixed", False):
                # PREFIX OPTIMIZATION: A single False routes the whole Parameter to static
                return False 
            
            # 3. Handle the dynamic routing
            if param_objects:
                # PREFIX OPTIMIZATION: A single True routes the whole Parameter to dynamic
                return True
            else:
                # SURGICAL MODE: Only .value is dynamic
                # We build a localized mask for just this parameter
                false_param = jax.tree_util.tree_map(lambda _: False, node)
                return eqx.tree_at(lambda p: p.value, false_param, True)
        
        # 4. If the node is a standard leaf (e.g., a non-parameter array), default to static
        return False

    # Map our mask builder across the tree, intercepting at Parameters
    filter_spec = jax.tree_util.tree_map(build_mask, pytree, is_leaf=is_valid_param)
    
    # Partition once using the fully constructed, bug-free boolean tree
    return eqx.partition(pytree, filter_spec)