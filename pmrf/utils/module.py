import jax.numpy as jnp
import parax as prx

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
        u = group.distribution.cdf(x)
        
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
    groups = module.param_groups(include_fixed=False)
    flat_vals = module.named_flat_param_values(include_fixed=False)
    
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
        x = group.distribution.icdf(u)
        
        if len(arrays) == 1:
            new_vals[group.param_names[0]] = x
        else:
            for i, name in enumerate(group.param_names):
                new_vals[name] = x[i]
                
    return module.with_params(new_vals)

def module_log_prob(module: prx.Module):
    """
    Calculates the total summed log probability of the module's parameters
    based on their assigned distributions.
    """
    groups = module.param_groups(include_fixed=False)
    flat_vals = module.named_flat_param_values(include_fixed=False)
    
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