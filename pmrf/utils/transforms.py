import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Any, Tuple, TypeVarTuple

from pmrf.utils.tree import unwrap, batch_axes

# Define a variadic type variable to capture the exact types of *args
Ts = TypeVarTuple("Ts")


def is_jax_array(x: Any) -> bool:
    return isinstance(x, jax.Array)


def is_inexact_jax_array(x):
    return is_jax_array(x) and jnp.issubdtype(x.dtype, jnp.inexact)


def derivative(eval_fn: Callable[..., Any], *args: *Ts, space: str = 'declared') -> Tuple[*Ts]:
    """
    Computes the exact derivative of a function.

    Dynamically routes to the most relevant autodiff method:
    - `grad` (reverse-mode) for scalar outputs.
    - `jacfwd` (forward-mode) for wide Jacobians (output size > input size).
    - `jacrev` (reverse-mode) for tall Jacobians (input size >= output size).

    Parameters are differentiated with respect to their value in `space`, so by
    default a derivative is per unit the parameter declares: per pF for a
    capacitance declared with ``scale=1e-12``. A tie is recomputed from its source
    inside the differentiated function, so the source's derivative includes the
    path through the tie. Raw JAX arrays are differentiated as they are.

    Safely handles models as inputs by filtering out non-differentiable
    static fields (like strings, booleans, integers, AND NumPy arrays).

    Parameters
    ----------
    eval_fn : Callable
        The function to differentiate. It is called with the arguments unwrapped.
    *args : *Ts
        The arguments to evaluate the derivative at. Can be raw arrays or full Models.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space parameter values are differentiated in. See :class:`pmrf.Param`.

    Returns
    -------
    Tuple[*Ts]
        A tuple containing the derivatives, one per argument. Each keeps its
        argument's structure with every parameter replaced by its derivative, so
        :func:`pmrf.param_values` reads the derivatives by name. Non-differentiable
        leaves are ``None``.

    Examples
    --------
    Compute the sensitivity of a component's response with respect to its parameters:

    >>> import pmrf as prf
    >>> from pmrf.models import ShuntCapacitor
    >>>
    >>> freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
    >>> cap = ShuntCapacitor(C=prf.Unconstrained(1.0, scale=1e-12), name='c1')
    >>>
    >>> def eval_s21(model):
    ...     return model.s_mag(freq)[0, 1, 0]
    ...
    >>> (d_cap,) = derivative(eval_s21, cap)
    >>>
    >>> # Per pF, the unit the capacitance was declared in
    >>> print(f"{prf.param_values(d_cap)['C']:.3e}")
    -1.164e-01
    """
    # Imported here: pmrf.parameters imports pmrf.utils.
    from pmrf.parameters import _check_space, _read, _set_paths, _with_fixed, _write, tree_param_paths
    from pmrf.utils.tree import Pathgetter

    _check_space(space)

    # Each parameter is replaced by its value in `space`, and rebuilt from it inside
    # the differentiated function, so the derivative is taken with respect to that
    # value and the chain rule carries it through scale, constraint and ties. Fixed
    # is an optimisation state, not a zero sensitivity, so parameters are rebuilt free.
    rebuilds = []
    surrogates = []
    for arg in args:
        resolved = [
            (path, node) for path, node in tree_param_paths(arg).values()
            if not isinstance(node, jax.Array)
        ]
        paths = [path for path, _ in resolved]
        nodes = [node for _, node in resolved]
        rebuilds.append((paths, nodes))
        surrogates.append(_set_paths(arg, paths, [_read(node, space) for node in nodes]))

    def _rebuild(surrogate_args):
        rebuilt = []
        for surrogate, (paths, nodes) in zip(surrogate_args, rebuilds):
            values = [Pathgetter(path)(surrogate) for path in paths]
            rebuilt.append(_set_paths(
                surrogate, paths, [_with_fixed(_write(node, value, space), False) for node, value in zip(nodes, values)]
            ))
        return unwrap(tuple(rebuilt))

    dynamic, static = eqx.partition(tuple(surrogates), is_inexact_jax_array)

    def _wrapper(dyn):
        args_tuple = _rebuild(eqx.combine(dyn, static))
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
    grid: bool = False,
    template: Any = None
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
        The arguments to evaluate. Can be raw arrays or full PyTree Models.
    grid : bool, default=False
        If False, performs a standard parallel sweep (zip-like) over the leading 
        dimension of all dynamic arrays in `args`.
        If True, computes the cartesian product (meshgrid) of the arguments.
    template : Any, optional
        An unbatched PyTree (or tuple of PyTrees mirroring `*args`) used as a 
        structural template to automatically extract batch axes. Required when 
        sweeping over complex, pre-batched PyTrees (like Bayesian samples).
        Cannot be used concurrently with `grid=True`.

    Returns
    -------
    Any
        The batched output of `eval_fn`.
    """
    args = unwrap(args)

    # ---------------------------------------------------------
    # MODE 1: Pre-Batched PyTrees (e.g. Bayesian Samples)
    # ---------------------------------------------------------
    if template is not None:
        if grid:
            raise ValueError("`grid=True` is not supported when `template` is provided.")
        
        # Normalize template to a tuple matching *args
        templates = template if isinstance(template, tuple) else (template,)
        if len(templates) != len(args):
            raise ValueError(f"Expected {len(args)} templates to match arguments, got {len(templates)}.")
        
        # Extract batch axes dynamically and execute
        in_axes = tuple(batch_axes(arg, temp) for arg, temp in zip(args, templates))
        return eqx.filter_vmap(eval_fn, in_axes=in_axes)(*args)

    # ---------------------------------------------------------
    # MODE 2: Standard Parallel Array Sweeps
    # ---------------------------------------------------------
    if not grid:
        dynamic, static = eqx.partition(args, is_jax_array)
        def _wrapper(dyn):
            args_tuple = eqx.combine(dyn, static)
            return eval_fn(*args_tuple)
        return jax.vmap(_wrapper)(dynamic)

    # ---------------------------------------------------------
    # MODE 3: Cartesian Grid Array Sweeps
    # ---------------------------------------------------------
    sizes = []
    for arg in args:
        dyn, _ = eqx.partition(arg, is_jax_array)
        leaves = jax.tree.leaves(dyn)
        if leaves:
            sizes.append(leaves[0].shape[0])
            
    if not sizes:
        return eval_fn(*args)

    mesh_indices = jnp.meshgrid(*[jnp.arange(s) for s in sizes], indexing="ij")
    flat_indices = [m.flatten() for m in mesh_indices]

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

    flat_dyn, flat_stat = eqx.partition(tuple(flat_args), is_jax_array)
    def _flat_wrapper(dyn):
        args_tuple = eqx.combine(dyn, flat_stat)
        return eval_fn(*args_tuple)
    
    flat_out = jax.vmap(_flat_wrapper)(flat_dyn)
    grid_shape = tuple(sizes)

    def _reshape_to_grid(x):
        return x.reshape(grid_shape + x.shape[1:]) if is_jax_array(x) else x

    out_dynamic, out_static = eqx.partition(flat_out, is_jax_array)
    out_dynamic_reshaped = jax.tree.map(_reshape_to_grid, out_dynamic)
    
    return eqx.combine(out_dynamic_reshaped, out_static)