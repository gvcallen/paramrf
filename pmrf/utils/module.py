from typing import Any, Literal
import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from parax.utils.distribution import (
    hypercube_to_physical as hypercube_to_physical, 
    physical_to_hypercube as physical_to_hypercube
)

from parax.parameter import Parameter, is_free_param
from parax.tree import partition
from parax.module import Module

# ------------------------------------------------------------------
# Bounds Extraction (Element-wise safe)
# ------------------------------------------------------------------

def make_bounds(
    tree: PyTree, 
    lower_override: float | None = None, 
    upper_override: float | None = None
) -> tuple[PyTree, PyTree]:
    """
    Extract the lower and upper bounds of all free parameters in a PyTree.

    This function traverses a PyTree and returns two structurally identical PyTrees
    containing only the lower and upper bounds for each `Parameter`. Because bounds
    are strictly element-wise properties, this operation safely ignores multivariate
    groupings.

    Parameters
    ----------
    tree : PyTree
        The input PyTree containing parax Parameters.
    lower_override : float or None, optional
        A global scalar to override all lower bounds. If None, falls back to the
        parameter's defined bounds, then its ICDF limit, and finally -inf.
    upper_override : float or None, optional
        A global scalar to override all upper bounds. If None, falls back to the
        parameter's defined bounds, then its ICDF limit, and finally inf.

    Returns
    -------
    tuple of (PyTree, PyTree)
        A tuple `(lower_bounds, upper_bounds)` containing PyTrees of the extracted
        boundary arrays. Static/fixed parameters are partitioned out.
    """
    def get_lower(x: Parameter):
        if not is_free_param(x): return x
        if lower_override is not None: return x.with_value(jnp.full_like(x.value, lower_override))
        if x.bounds is not None: return x.with_value(x.bounds[..., 0])
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.with_value(x.distribution.icdf(jnp.full_like(x.value, 0.001))) 
        return x.with_value(jnp.full_like(x.value, -jnp.inf))

    def get_upper(x: Parameter):
        if not is_free_param(x): return x
        if upper_override is not None: return x.with_value(jnp.full_like(x.value, upper_override))
        if x.bounds is not None: return x.with_value(x.bounds[..., 1])
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.with_value(x.distribution.icdf(jnp.full_like(x.value, 1.0 - 0.001))) 
        return x.with_value(jnp.full_like(x.value, jnp.inf))

    lower_tree = jax.tree.map(get_lower, tree, is_leaf=is_free_param)
    upper_tree = jax.tree.map(get_upper, tree, is_leaf=is_free_param)

    (lower_bounds, upper_bounds), _ = partition((lower_tree, upper_tree))
    return lower_bounds, upper_bounds

# ------------------------------------------------------------------
# Spatial Transformations (Group-aware for Multivariate)
# ------------------------------------------------------------------

def physical_to_hypercube(tree: PyTree) -> Any:
    """
    Transform all free parameters in a PyTree from the physical domain to the hypercube.

    This maps parameters into the unit space [0, 1] using their respective Cumulative 
    Distribution Functions (CDFs). If the input tree is a `parax.Module`, this safely 
    handles multivariate distributions by temporarily grouping the parameters. 
    The unit mapped values are injected directly into the parameter's `latent_value`.

    Parameters
    ----------
    tree : PyTree
        The input PyTree containing parax Parameters.

    Returns
    -------
    PyTree
        A structurally identical PyTree where all free parameters with defined
        distributions have their `latent_value` updated to the [0, 1] hypercube.
    """
    # PATH A: Group-Aware Mapping (Safe for Multivariate spanning multiple leaves)
    if isinstance(tree, Module):
        groups = tree.param_groups(include_fixed=False)
        flat_vals = tree.named_flat_param_values(include_fixed=False)
        flat_params = tree.named_flat_params(include_fixed=False) # Need full objects to inject latent
        
        new_params = {}
        for group in groups:
            if group.distribution is None:
                for name in group.param_names:
                    new_params[name] = flat_params[name]
                continue
                
            arrays = [flat_vals[name] for name in group.param_names]
            x = jnp.stack(arrays)
            if len(arrays) == 1:
                x = jnp.squeeze(x, axis=0)
                
            # Map physical to hypercube [0, 1]
            u = physical_to_hypercube(group.distribution, x)
            
            # Crucial Fix: We must inject 'u' into `latent_value` directly. 
            # We do NOT want to pass 'u' into `Parameter(value=u)` because that 
            # would apply the transform's inverse to the hypercube state.
            if len(arrays) == 1:
                new_params[group.param_names[0]] = dataclasses.replace(flat_params[group.param_names[0]], latent_value=u)
            else:
                for i, name in enumerate(group.param_names):
                    new_params[name] = dataclasses.replace(flat_params[name], latent_value=u[i])
                    
        return tree.with_params(new_params)

    # PATH B: Naive Tree Mapping (Requires Multivariate params to be single arrays)
    def _map_to_u(x: Parameter):
        if not is_free_param(x) or x.distribution is None: return x
        u = physical_to_hypercube(x.distribution, x.value)
        return dataclasses.replace(x, latent_value=u)

    return jax.tree.map(_map_to_u, tree, is_leaf=is_free_param)


def hypercube_to_physical(tree: PyTree) -> Any:
    """
    Transform all free parameters in a PyTree from the hypercube back to physical space.

    This maps parameters from the [0, 1] unit space back to their original physical 
    bounds using the Inverse Cumulative Distribution Function (ICDF). It reads the 
    current state from the parameter's `latent_value` (assumed to be in the hypercube)
    and updates the parameter safely, restoring the proper unconstrained `latent_value` 
    via the underlying bijector inverse.

    Parameters
    ----------
    tree : PyTree
        The input PyTree containing parax Parameters operating in the hypercube.

    Returns
    -------
    PyTree
        A structurally identical PyTree with parameters restored to their physical values.
    """
    # PATH A: Group-Aware Mapping
    if isinstance(tree, Module):
        groups = tree.param_groups(include_fixed=False)
        # Note: We must extract latent_value here, because the optimizer has been updating it in U-space
        flat_params = tree.named_flat_params(include_fixed=False)
        flat_latents = {k: v.latent_value for k, v in flat_params.items()}
        
        new_params = {}
        for group in groups:
            if group.distribution is None:
                for name in group.param_names:
                    new_params[name] = flat_params[name]
                continue
                
            arrays = [flat_latents[name] for name in group.param_names]
            u = jnp.stack(arrays)
            if len(arrays) == 1:
                u = jnp.squeeze(u, axis=0)
                
            # Map from hypercube [0, 1] back to physical values
            x = hypercube_to_physical(group.distribution, u)
            
            # Safe to use `.with_value(x)` here. It will calculate the correct unconstrained 
            # latent_value automatically using the bijector's inverse.
            if len(arrays) == 1:
                new_params[group.param_names[0]] = flat_params[group.param_names[0]].with_value(x)
            else:
                for i, name in enumerate(group.param_names):
                    new_params[name] = flat_params[name].with_value(x[i])
                    
        return tree.with_params(new_params)

    # PATH B: Naive Tree Mapping
    def _map_to_x(p: Parameter):
        if not is_free_param(p) or p.distribution is None: return p
        u = p.latent_value
        x = hypercube_to_physical(p.distribution, u)
        return p.with_value(x)

    return jax.tree.map(_map_to_x, tree, is_leaf=is_free_param)


def enforce_bounds(module: PyTree) -> PyTree:
    """
    Apply auto-generated bounding bijectors to free parameters.

    This maps over the tree and dynamically constructs `ScalarAffine`, `Sigmoid`, 
    or `Softplus` bijector chains to enforce parameter bounds mapping to the real 
    line. Parameters with pre-existing transforms are safely bypassed, respecting 
    the user's explicit architectural mapping.

    Parameters
    ----------
    module : PyTree
        The input PyTree containing parax Parameters.

    Returns
    -------
    PyTree
        A structurally identical PyTree where unbounded free parameters have 
        been updated with appropriate bounds-enforcing bijectors.
    """
    from distreqx.bijectors import Chain, Sigmoid, Softplus, ScalarAffine

    def _apply_bounding_bijectors(x: Parameter):
        if not is_free_param(x) or x.bounds is None or x.transform is not None: 
            return x
        
        lower = x.bounds[..., 0]
        upper = x.bounds[..., 1]
        
        has_lower = jnp.any(lower > -jnp.inf)
        has_upper = jnp.any(upper < jnp.inf)
        
        if not has_lower and not has_upper:
            return x
        
        # Construct the correct mapping to the real line using ScalarAffine
        # Chain applies right-to-left: e.g., ScalarAffine(Sigmoid(x))
        if has_lower and has_upper:
            scale = upper - lower
            bijector = Chain([ScalarAffine(shift=lower, scale=scale), Sigmoid()])
        elif has_lower and not has_upper:
            # Scale defaults to 1.0 if not provided
            bijector = Chain([ScalarAffine(shift=lower), Softplus()])
        elif not has_lower and has_upper:
            bijector = Chain([ScalarAffine(shift=upper, scale=jnp.full_like(upper, -1.0)), Softplus()])
            
        # Re-initialize the parameter with the new transform
        return x.with_transform(transform=bijector)

    return jax.tree.map(_apply_bounding_bijectors, module, is_leaf=is_free_param)


def log_prob(module):
    """
    Calculates the total summed log probability of the module's parameters
    based on their assigned distributions.
    """
    groups = module.param_groups()
    flat_vals = module.named_flat_param_values()
    
    total_log_prob = 0.0
    for group in groups:
        if group.distribution is None:
            continue
            
        arrays = [flat_vals[name] for name in group.param_names]
        x = jnp.stack(arrays)
        
        if len(arrays) == 1:
            x = jnp.squeeze(x, axis=0)
            
        # Calculate log_prob for the group and accumulate
        total_log_prob = total_log_prob + jnp.sum(group.distribution.log_prob(x))
        
    return total_log_prob