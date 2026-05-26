import jax
import jax.numpy as jnp
import equinox as eqx

def error_if(x, pred, msg, *print_args, on_error="default", **print_kwargs):
    """
    Conditionally halts JAX execution and prints formatted debug values at runtime.

    This function evaluates a boolean condition inside JIT-compiled code. If the 
    condition is met (e.g., an unphysical parameter like negative impedance is 
    detected), it outputs the formatted runtime arrays to standard output before 
    terminating the program. The input data `x` is threaded through the control 
    flow to enforce strict execution order within the XLA graph.

    Parameters
    ----------
    x : Any
        The input data (JAX array or PyTree) to pass through. This is required 
        to maintain computational dependencies in the compiled graph.
    pred : bool or jax.Array
        A boolean condition or boolean array. If any element evaluates to `True`, 
        execution halts and the debug message is printed.
    msg : str
        The format string for the error message and console output. Uses standard 
        Python `{}` formatting to dynamically inject JAX arrays.
    *print_args : Any
        Dynamic tensors or values to sequentially format into the `msg` string.
    on_error : str, optional
        The internal error handling mode. Default is "default".
    **print_kwargs : Any
        Dynamic tensors or values to format into the `msg` string via keywords.

    Returns
    -------
    Any
        The unmodified input `x`.

    Raises
    ------
    RuntimeError
        Triggered if `pred` contains any `True` elements during runtime.

    Examples
    --------
    Catching an invalid characteristic impedance inside a JIT-compiled simulation:

    >>> z0 = jnp.array(-50.0)
    >>> is_invalid = z0 < 0
    >>> z0 = error_if(
    ...     z0, 
    ...     is_invalid, 
    ...     "Characteristic impedance must be positive, got Z0 = {} Ohms", 
    ...     z0
    ... )
    """
    pred_scalar = jnp.any(pred)
    
    def print_and_pass():
        jax.debug.print(msg, *print_args, **print_kwargs)
        return x
        
    def just_pass():
        return x

    x_after_cond = jax.lax.cond(pred_scalar, print_and_pass, just_pass)
    
    return eqx.error_if(
        x_after_cond, 
        pred, 
        f"{msg} (Check standard output for runtime values)", 
        on_error=on_error
    )