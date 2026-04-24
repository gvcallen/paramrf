import jax
import jax.numpy as jnp
import parax as prx

from pmrf.utils.distribution import hypercube_to_physical as dist_hypercube_to_physical, physical_to_hypercube as dist_physical_to_hypercube

def make_bounds(
    module: prx.Module, 
    lower_override: float | None = None, 
    upper_override: float | None = None
) -> tuple[prx.Module, prx.Module]:
    def get_lower(x: prx.Parameter):
        if not prx.is_free_param(x):
            return x
            
        if lower_override is not None:
            return x.with_value(jnp.full_like(x.value, lower_override))
            
        if x.bounds is not None:
            return x.with_value(x.bounds[..., 0])
            
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.with_value(x.distribution.icdf(jnp.full_like(x.value, 0.001))) # TODO deprecate
            
        return x.with_value(jnp.full_like(x.value, -jnp.inf))

    def get_upper(x: prx.Parameter):
        if not prx.is_free_param(x):
            return x
            
        if upper_override is not None:
            return x.with_value(jnp.full_like(x.value, upper_override))
            
        if x.bounds is not None:
            return x.with_value(x.bounds[..., 1])
            
        if x.distribution is not None and hasattr(x.distribution, 'icdf'):
            return x.with_value(x.distribution.icdf(jnp.full_like(x.value, 1.0 - 0.001))) # TODO deprecate
            
        return x.with_value(jnp.full_like(x.value, jnp.inf))

    # 1. Map the respective bound functions over the module tree
    lower_tree = jax.tree.map(get_lower, module, is_leaf=prx.is_free_param)
    upper_tree = jax.tree.map(get_upper, module, is_leaf=prx.is_free_param)

    # 2. Partition to extract just the bounding parameters for the solver
    (lower_bounds, upper_bounds), _ = prx.partition((lower_tree, upper_tree))
    
    return lower_bounds, upper_bounds  

def physical_to_hypercube(module: prx.Module):
    """
    Transforms the module's parameters from their physical domain 
    to the [0, 1] hypercube using the cumulative distribution function (CDF).
    """
    groups = module.param_groups(include_fixed=False)
    flat_vals = module.named_flat_param_values(include_fixed=False)
    
    new_vals = {}
    for group in groups:
        # If no distribution is defined, retain the original values
        if group.distribution is None:
            for name in group.param_names:
                new_vals[name] = flat_vals[name]
            continue
            
        # Extract and stack arrays for the group
        arrays = [flat_vals[name] for name in group.param_names]
        x = jnp.stack(arrays)
        
        # Squeeze scalar parameters back to shape () for Univariate distributions
        if len(arrays) == 1:
            x = jnp.squeeze(x, axis=0)
            
        # Map to hypercube [0, 1]
        u = dist_physical_to_hypercube(group.distribution, x)
        
        # Unpack back into the flat dictionary
        if len(arrays) == 1:
            new_vals[group.param_names[0]] = u
        else:
            for i, name in enumerate(group.param_names):
                new_vals[name] = u[i]
                
    return module.with_params(new_vals)

def hypercube_to_physical(module: prx.Module):
    """
    Transforms the module's parameters from the [0, 1] hypercube 
    back to their physical domain using the inverse CDF (icdf).
    """
    groups = module.param_groups()
    flat_vals = module.named_flat_param_values()
    
    new_vals = {}
    for group in groups:
        if group.distribution is None:
            for name in group.param_names:
                new_vals[name] = flat_vals[name]
            continue
            
        arrays = [flat_vals[name] for name in group.param_names]
        u = jnp.stack(arrays)
        
        if len(arrays) == 1:
            u = jnp.squeeze(u, axis=0)
            
        # Map from hypercube [0, 1] back to physical values
        x = dist_hypercube_to_physical(group.distribution, u)
        x = group.distribution.icdf(u)
        
        if len(arrays) == 1:
            new_vals[group.param_names[0]] = x
        else:
            for i, name in enumerate(group.param_names):
                new_vals[name] = x[i]
                
    return module.with_params(new_vals)

def log_prob(module: prx.Module):
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