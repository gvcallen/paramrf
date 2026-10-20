import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap


@functools.lru_cache(maxsize=None)
def _checked_predicate(msg, kw_names, npos, arg_tree, nbatch):
    """Print only failing values, retaining batch axes outside the decision."""

    @custom_vmap
    def checked(pred, args):
        def fail(_):
            def host_print(host_pred, *host_args):
                for index in np.ndindex(host_pred.shape[:nbatch]):
                    if np.any(host_pred[index]):
                        values = jax.tree_util.tree_unflatten(
                            arg_tree, [arg[index] for arg in host_args]
                        )
                        print(msg.format(*values[:npos], **dict(zip(kw_names, values[npos:]))))
                return host_pred

            return jax.pure_callback(
                host_print, jax.ShapeDtypeStruct(pred.shape, pred.dtype), pred, *args
            )

        return jax.lax.cond(jnp.any(pred), fail, lambda _: pred, None)

    @checked.def_vmap
    def rule(axis_size, in_batched, pred, args):
        pred_batched, args_batched = in_batched

        def broadcast(value, batched):
            return value if batched else jnp.broadcast_to(value, (axis_size, *value.shape))

        pred = broadcast(pred, pred_batched)
        args = tuple(broadcast(arg, batched) for arg, batched in zip(args, args_batched))
        return _checked_predicate(msg, kw_names, npos, arg_tree, nbatch + 1)(pred, args), True

    return checked


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
        pred = jnp.asarray(pred)
        kw_names = tuple(print_kwargs)
        args, arg_tree = jax.tree_util.tree_flatten((*print_args, *print_kwargs.values()))
        args = tuple(jnp.asarray(arg) for arg in jax.lax.stop_gradient(args))
        pred = _checked_predicate(msg, kw_names, len(print_args), arg_tree, 0)(pred, args)
        error_msg = f"{msg} (Check standard output for runtime values)"

    return eqx.error_if(x, pred, error_msg, on_error=on_error)
