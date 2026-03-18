import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.distributions import Distribution

from pmrf.utils.string import format_val
from pmrf.distributions import StackedDistribution

def split_vectorized_distribution(d: Distribution) -> list[Distribution]:
    """
    Split an arbitrarily shaped batch of univariate numpyro distributions into a list of scalar distributions.
    
    Handles broadcasting of distribution parameters to the batch shape.
    
    Parameters
    ----------
    d : numpyro.distributions.Distribution
        A distribution with ``event_shape == ()`` and arbitrary ``batch_shape``.

    Returns
    -------
    list[numpyro.distributions.Distribution]
        A flat list of scalar distributions corresponding to the flattened batch.
    """
    if d.event_shape != ():
        raise ValueError(f"Cannot split distribution with event_shape={d.event_shape} (likely an Independent or Multivariate dist)")

    batch_shape = d.batch_shape
    if not batch_shape: # Scalar distribution
        return [d]

    # Calculate total size to verify flatten length
    total_size = 1
    for dim in batch_shape:
        total_size *= dim
        
    # Special handling for ImproperUniform
    if isinstance(d, dist.ImproperUniform):
        return [
            dist.ImproperUniform(d.support, batch_shape=(), event_shape=d.event_shape)
            for _ in range(total_size)
        ]        

    # Get all init params used to construct the distribution (e.g., 'loc', 'scale', 'low', 'high')
    # d.arg_constraints keys usually match the __init__ arguments
    param_names = d.arg_constraints.keys()
    
    # Extract current values of parameters
    param_values = {name: getattr(d, name) for name in param_names}

    # Broadcast all parameters to the distribution's batch_shape and flatten them
    flat_params = {}
    for name, val in param_values.items():
        val = jnp.asarray(val)
        # Broadcast the parameter value to the distribution's batch shape.
        # This handles cases where e.g. Normal(0, [1, 2]) has scalar loc and vector scale.
        try:
            val_broadcast = jnp.broadcast_to(val, batch_shape)
        except ValueError:
             # Fallback or error if shapes are strictly incompatible, though numpyro usually prevents this earlier
             raise ValueError(f"Parameter '{name}' with shape {val.shape} cannot be broadcast to batch_shape {batch_shape}")
             
        flat_params[name] = jnp.ravel(val_broadcast)

    # Reconstruct individual scalar distributions
    split_dists = []
    dist_class = d.__class__
    
    for i in range(total_size):
        # Extract the i-th scalar value for each parameter
        args = {name: vals[i] for name, vals in flat_params.items()}
        split_dists.append(dist_class(**args))

    return split_dists

def stack_vectorized_distributions(dists: list[Distribution | None]) -> Distribution | None:
    """
    Combine a list of scalar numpyro distributions into a single batched distribution.
    
    Parameters
    ----------
    dists : list[numpyro.distributions.Distribution | None]
        A list of distributions to stack.

    Returns
    -------
    numpyro.distributions.Distribution | None
        The vectorized distribution, or None if no distributions were provided.
    """
    if all(d is None for d in dists):
        return None
    if any(d is None for d in dists):
        raise ValueError("Cannot stack a mix of parameters where some have distributions and others do not.")
        
    dist_classes = set(type(d) for d in dists)
    
    # Fast path: If they are all the exact same family, use native NumPyro batching
    if len(dist_classes) == 1:
        dist_cls = dists[0].__class__
        param_names = dists[0].arg_constraints.keys()
        stacked_kwargs = {name: jnp.stack([getattr(d, name) for d in dists]) for name in param_names}
        return dist_cls(**stacked_kwargs)
        
    # Flexible path: Use our custom meta-distribution for mixed types!
    return StackedDistribution(dists)

def serialize_distribution(d: Distribution | None) -> dict | None:
    r"""
    Serialize a numpyro distribution to a lightweight dictionary.

    Parameters
    ----------
    d : numpyro.distributions.Distribution or None
        The distribution to serialize.

    Returns
    -------
    dict or None
        A dictionary with ``class`` and ``params`` keys, or ``None``.
    """
    if d is None:
        return None
        
    params = {}
    for k, v in d.__dict__.items():
        if k.startswith("_"):
            continue
        if isinstance(v, jnp.ndarray):
            params[k] = v.tolist()
        elif isinstance(v, dist.constraints.Constraint):
            # Just store the name of the constraint (e.g., 'real', 'positive')
            params[k] = {"__constraint__": type(v).__name__}
        else:
            params[k] = v
            
    return {"class": d.__class__.__name__, "params": params}

# Helper to deserialize a numpyro Distribution
def deserialize_distribution(dct: dict | None) -> Distribution | None:
    r"""
    Deserialize a numpyro distribution from a dictionary.

    Parameters
    ----------
    dct : dict or None
        A dictionary produced by :func:`_serialize_distribution`.

    Returns
    -------
    numpyro.distributions.Distribution or None
        The reconstructed distribution, or ``None``.
    
    Raises
    ------
    ValueError
        If the distribution class is unknown.
    """
    if dct is None:
        return None
        
    cls = getattr(dist, dct["class"], None)
    if cls is None:
        raise ValueError(f"Unknown distribution class: {dct['class']}")
        
    params = dct["params"]
    for k, v in params.items():
        if isinstance(v, dict) and "__constraint__" in v:
            # Map the string name back to the numpyro constraint object
            constraint_name = v["__constraint__"].lower() # e.g. '_Real' -> 'real'
            constraint_name = constraint_name.strip('_') 
            params[k] = getattr(dist.constraints, constraint_name)
            
    return cls(**params)


def format_distribution(d: Distribution) -> str:
    """Format a numpyro distribution dynamically using its arg_constraints."""
    class_name = d.__class__.__name__
    if hasattr(d, "arg_constraints"):
        args = []
        # Pull out loc, scale, low, high, etc., dynamically
        for param_name in d.arg_constraints.keys():
            if hasattr(d, param_name):
                val = getattr(d, param_name)
                args.append(f"{param_name}={format_val(val)}")
        return f"{class_name}({', '.join(args)})"
    return repr(d)