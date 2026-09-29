import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def error_if(x, pred, msg, *print_args, on_error="default", **print_kwargs):
    """
    Conditionally halts JAX execution and prints formatted debug values at runtime.

    This function evaluates a boolean condition inside JIT-compiled code. If the 
    condition is met (e.g., an unphysical parameter like negative impedance is 
    detected), it outputs the formatted runtime arrays to standard output before 
    raising. Whether to print is decided on the host, so nothing is printed when
    the check passes, under any transform. Under `jax.vmap`, the message is printed
    once for each failing batch element, with that element's values. Without print
    arguments nothing is printed: the raised error carries `msg`.

    Parameters
    ----------
    x : Any
        The input data (JAX array or PyTree) to pass through. The check runs only
        if the returned value is used.
    pred : bool or jax.Array
        A boolean condition or boolean array. If any element evaluates to `True`,
        execution halts, printing the message if print arguments are given.
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
    equinox.EquinoxRuntimeError
        At runtime, if `pred` contains any `True` elements.

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
    error_msg = msg
    if print_args or print_kwargs:
        # Decided on the host: a `lax.cond` on a batched predicate lowers to `select`
        # under `vmap`, which runs the print whether or not the check fails. A
        # `pure_callback` rather than `jax.debug.callback`, so that `eqx.error_if` can
        # consume its output: that data dependency makes the print precede the raise.
        def print_if(host_pred, args, kwargs):
            if np.any(host_pred):
                print(msg.format(*args, **kwargs))
            return host_pred

        pred = jnp.asarray(pred)
        args, kwargs = jax.lax.stop_gradient((print_args, print_kwargs))
        pred = jax.pure_callback(
            print_if,
            jax.ShapeDtypeStruct(pred.shape, pred.dtype),
            pred,
            args,
            kwargs,
            vmap_method="sequential",
        )
        error_msg = f"{msg} (Check standard output for runtime values)"

    return eqx.error_if(x, pred, error_msg, on_error=on_error)
