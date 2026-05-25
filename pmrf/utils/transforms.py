import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Any, Tuple, TypeVarTuple

from pmrf.utils.tree import unwrap

# Define a variadic type variable to capture the exact types of *args
Ts = TypeVarTuple("Ts")


def is_jax_array(x: Any) -> bool:
    return isinstance(x, jax.Array)


def is_inexact_jax_array(x):
    return is_jax_array(x) and jnp.issubdtype(x.dtype, jnp.inexact)


def derivative(eval_fn: Callable[..., Any], *args: *Ts) -> Tuple[*Ts]:
    """
    Computes the exact derivative of a function.

    Dynamically routes to the most relevant autodiff method:
    - `grad` (reverse-mode) for scalar outputs.
    - `jacfwd` (forward-mode) for wide Jacobians (output size > input size).
    - `jacrev` (reverse-mode) for tall Jacobians (input size >= output size).

    Safely handles models as inputs by filtering out non-differentiable 
    static fields (like strings, booleans, integers, AND NumPy arrays).

    Parameters
    ----------
    eval_fn : Callable
        The function to differentiate.
    *args : *Ts
        The arguments to evaluate the derivative at. Can be raw arrays or full Models.

    Returns
    -------
    Tuple[*Ts]
        A tuple containing the derivatives. The tuple length and contents will 
        exactly mirror the types and structure of the input `args`.

    Examples
    --------
    Compute the sensitivity of a component's response with respect to its parameters:

    >>> import pmrf as prf
    >>> from pmrf.models import ShuntCapacitor
    >>>
    >>> freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    >>> cap = ShuntCapacitor(C=prf.Unconstrained(1.0e-12), name='c1')
    >>>
    >>> def eval_s21(model):
    ...     return model.s_mag(freq)[0, 1, 0]
    ...
    >>> (d_cap,) = derivative(eval_s21, cap)
    >>> 
    >>> # The structural layout of the model is preserved in the derivative
    >>> print(f"{d_cap.at('c1.C').get():.3e}")
    -1.060e-01
    """
    args = unwrap(args)

    dynamic, static = eqx.partition(args, is_inexact_jax_array)

    def _wrapper(dyn):
        args_tuple = eqx.combine(dyn, static)
        return eval_fn(*args_tuple)

    out_shape = jax.eval_shape(_wrapper, dynamic)
    leaves = jax.tree.leaves(out_shape)
    is_scalar = len(leaves) == 1 and getattr(leaves[0], "shape", None) == ()

    if is_scalar:
        return jax.grad(_wrapper)(dynamic)

    def _dynamic_size(tree):
        return sum(
            l.size for l in jax.tree.leaves(tree) 
            if is_inexact_jax_array(l)
        )

    in_size = _dynamic_size(dynamic)
    out_size = _dynamic_size(out_shape)

    if out_size > in_size:
        return jax.jacfwd(_wrapper)(dynamic)
    else:
        return jax.jacrev(_wrapper)(dynamic)


def sweep(
    eval_fn: Callable[..., Any], 
    *args: Any, 
    grid: bool = False
) -> Any:
    """
    Vectorizes a function over the given input arguments.

    Safely handles models as input by filtering out non-vmappable 
    static fields (like strings, objects, and NumPy arrays).

    Parameters
    ----------
    eval_fn : Callable
        The function to run the sweep over.
    *args : Any
        The arguments to evaluate. Can be raw arrays or full Models.
    grid : bool, default=False
        If False, performs a standard parallel sweep (zip-like) over the leading 
        dimension of all dynamic arrays in `args`. All swept arrays must have the 
        same leading dimension size.
        If True, computes the cartesian product (meshgrid) of the arguments, 
        evaluating the function at every possible combination of inputs.

    Returns
    -------
    Any
        The batched output of `eval_fn`. If `grid=True`, the output shape will 
        reflect the dimensional sizes of all swept inputs appended together.

    Examples
    --------
    Sweep component values to evaluate network responses across parameter spaces:

    >>> import jax.numpy as jnp
    >>> import pmrf as prf
    >>> from pmrf.models import ShuntCapacitor, Inductor
    >>>
    >>> freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    >>> c_vals = jnp.linspace(1e-12, 5e-12, 10)
    >>> l_vals = jnp.linspace(1e-9, 5e-9, 10)
    >>>
    >>> def eval_s21(c, l):
    ...     model = ShuntCapacitor(C=c) ** Inductor(L=l)
    ...     return model.s_mag(freq)[0, 1, 0]
    ...
    >>> # Standard parallel sweep (evaluates pairs index-by-index)
    >>> out_parallel = sweep(eval_s21, c_vals, l_vals)
    >>> out_parallel.shape
    (10,)
    >>>
    >>> # Grid sweep (computes the full Cartesian product)
    >>> out_grid = sweep(eval_s21, c_vals, l_vals, grid=True)
    >>> out_grid.shape
    (10, 10)
    """
    args = unwrap(args)

    if not grid:
        dynamic, static = eqx.partition(args, is_jax_array)
        def _wrapper(dyn):
            args_tuple = eqx.combine(dyn, static)
            return eval_fn(*args_tuple)
        return jax.vmap(_wrapper)(dynamic)

    # Determine the size of the leading dimension
    sizes = []
    for arg in args:
        dyn, _ = eqx.partition(arg, is_jax_array)
        leaves = jax.tree.leaves(dyn)
        if leaves:
            sizes.append(leaves[0].shape[0])
    if not sizes:
        return eval_fn(*args)

    # Create a grid of indices using 'ij' (matrix)
    mesh_indices = jnp.meshgrid(*[jnp.arange(s) for s in sizes], indexing="ij")
    flat_indices = [m.flatten() for m in mesh_indices]

    # Gather the flattened combinations
    flat_args = []
    idx_counter = 0
    for arg in args:
        dyn, stat = eqx.partition(arg, is_jax_array)
        leaves = jax.tree.leaves(dyn)
        if leaves:
            flat_idx = flat_indices[idx_counter]
            dyn_gathered = jax.tree.map(lambda x: x[flat_idx], dyn)
            flat_args.append(eqx.combine(dyn_gathered, stat))
            idx_counter += 1
        else:
            flat_args.append(arg)

    # Run native vmap over the flattened 1D combinations and reshape
    flat_dyn, flat_stat = eqx.partition(tuple(flat_args), is_jax_array)
    def _flat_wrapper(dyn):
        args_tuple = eqx.combine(dyn, flat_stat)
        return eval_fn(*args_tuple)
    flat_out = jax.vmap(_flat_wrapper)(flat_dyn)

    grid_shape = tuple(sizes)

    def _reshape_to_grid(x):
        if is_jax_array(x):
            return x.reshape(grid_shape + x.shape[1:])
        return x

    out_dynamic, out_static = eqx.partition(flat_out, is_jax_array)
    out_dynamic_reshaped = jax.tree.map(_reshape_to_grid, out_dynamic)
    
    return eqx.combine(out_dynamic_reshaped, out_static)