from typing import Sequence

import jax.numpy as jnp

from pmrf.parameters.parameter import Parameter, _stack_vectorized_distributions

def Fixed(value, n: int | None = None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` that is marked as fixed.

    Parameters
    ----------
    value
        The value of the parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created fixed Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        value = jnp.array(value)
    return Parameter(value=value, fixed=True, **kwargs)

def Free(value, n: int | None = None, **kwargs) -> Parameter:
    r"""
    Create a `Parameter` that is marked as not fixed (i.e., free to vary).

    Parameters
    ----------
    value
        The value of the parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created free Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        value = jnp.array(value)
    return Parameter(value=value, **kwargs)

def Stacked(parameters: Sequence[Parameter], name: str | None = None, **kwargs) -> Parameter:
    """
    Combine multiple scalar or identically-shaped Parameters into a single vectorized Parameter.
    
    This acts as the inverse of `Parameter.flattened()`.

    Parameters
    ----------
    parameters : Sequence[Parameter]
        The list/tuple of Parameter objects to stack.
    name : str, optional
        The overarching name for the new stacked parameter.
    **kwargs
        Additional arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        A new parameter containing the stacked values and distributions.
    """
    if not parameters:
        raise ValueError("Cannot stack an empty sequence of parameters.")
        
    # 1. Stack the unscaled values
    values = jnp.stack([p.value for p in parameters])
    
    # 2. Combine distributions
    dists = [p.distribution for p in parameters]
    stacked_dist = _stack_vectorized_distributions(dists)
    
    # 3. Preserve or generate flat names
    flat_names = []
    for p in parameters:
        if p.flat_names is not None:
            flat_names.extend(p.flat_names)
        elif p.name is not None:
            flat_names.append(p.name)
        else:
            flat_names.append(None)
            
    # 4. Handle the 'fixed' flag
    # Note: Parameter.size evaluates `if self.fixed:`, meaning `fixed` must remain a scalar bool.
    fixed_flags = [p.fixed for p in parameters]
    if not all(f == fixed_flags[0] for f in fixed_flags):
        raise ValueError(
            "All parameters must have the exact same 'fixed' status to be stacked. "
            "Element-wise fixed arrays are not supported by the base Parameter class."
        )
        
    # 5. Handle scales (we DON'T allow heterogeneous scales)
    scales = [p.scale for p in parameters]
    if not all(s == scales[0] for s in scales):
        raise Exception("Cannot create a stacked Parameter with differing scales")
    
    scale = scales[0]
    return Parameter(
        value=values,
        distribution=stacked_dist,
        fixed=fixed_flags[0],
        scale=scale,
        name=name,
        flat_names=flat_names,
        **kwargs
    )