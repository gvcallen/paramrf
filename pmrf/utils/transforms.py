import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Any, Tuple, TypeVarTuple

# Define a variadic type variable to capture the exact types of *args
Ts = TypeVarTuple("Ts")

def derivative(eval_fn: Callable[..., Any], *args: *Ts) -> Tuple[*Ts]:
    """
    Computes the exact analytical derivative of a function.

    Dynamically routes to the most computationally efficient autodiff backend:
    - `grad` (reverse-mode) for scalar outputs.
    - `jacfwd` (forward-mode) for wide Jacobians (output size > input size).
    - `jacrev` (reverse-mode) for tall Jacobians (input size >= output size).

    Safely handles PyTrees (like Models) by filtering out non-differentiable 
    static fields (like strings, booleans, or integers).

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
    """
    # Pack args into a single tuple to satisfy Equinox's single-argument rule
    def _wrapper(args_tuple):
        return eval_fn(*args_tuple)

    # Trace the function to inspect output shapes
    out_shape = eqx.filter_eval_shape(eval_fn, *args)
    
    # Check if the output is a single scalar (required for `grad`)
    leaves = jax.tree_util.tree_leaves(out_shape)
    is_scalar = len(leaves) == 1 and getattr(leaves[0], "shape", None) == ()

    if is_scalar:
        return eqx.filter_grad(_wrapper)(args)

    # For Jacobians, count inexact parameters to optimize the autodiff direction
    def _inexact_size(tree):
        return sum(
            l.size for l in jax.tree_util.tree_leaves(tree) 
            if eqx.is_inexact_array_like(l)
        )

    in_size = _inexact_size(args)
    out_size = _inexact_size(out_shape)

    # Route based on which pass requires fewer VJP/JVP evaluations
    if out_size > in_size:
        return eqx.filter_jacfwd(_wrapper)(args)
    else:
        return eqx.filter_jacrev(_wrapper)(args)
    

def sweep(eval_fn: Callable, model: Any, grid: bool = False, **sweeps) -> Any:
    """
    Evaluates a function over vectors of parameters.

    This engine structurally injects arrays into the model, bypassing 
    standard __init__ validation, allowing users to keep their class 
    constructors strictly scalar and pure.

    Parameters
    ----------
    eval_fn : Callable
        The simulation function to evaluate. Must accept the `model` as its 
        only argument and return a JAX array or PyTree of arrays.
    model : Any
        The nominal model instance to be evaluated.
    grid : bool, optional
        If True, computes the N-dimensional Cartesian product (a full grid sweep). 
        If False (default), computes a 1-to-1 element-wise sweep, where all 
        sweep arrays must have the identical length.
    **sweeps : ArrayLike
        Keyword arguments mapping parameter names/paths to JAX arrays.

    Returns
    -------
    Any
        The batched results. If `grid=True`, the output arrays will be prepended 
        with the lengths of all input arrays: `(len(a1), len(a2), ..., *out_shape)`.
        If `grid=False`, the output arrays will be `(len(a1), *out_shape)`.
    """
    if not sweeps:
        return eval_fn(model)

    keys = list(sweeps.keys())
    arrays = [jnp.asarray(sweeps[k]) for k in keys]
    
    # 1. Handle Cartesian product generation
    if grid:
        # Create N-dimensional grid and flatten for a single vmap pass
        grids = jnp.meshgrid(*arrays, indexing='ij')
        flat_arrays = [g.flatten() for g in grids]
        sweep_data = dict(zip(keys, flat_arrays))
        out_shape_prefix = tuple(len(a) for a in arrays)
    else:
        sweep_data = dict(zip(keys, arrays))
        out_shape_prefix = (len(arrays[0]),)

    # 2. Structural Injection (Bypasses __init__ validation)
    batched_model = model
    for path, values in sweep_data.items():
        batched_model = batched_model.at(path).set(values)

    # 3. Vectorized Execution
    # filter_vmap safely maps over array leaves while broadcasting static data (like strings)
    results = eqx.filter_vmap(eval_fn)(batched_model)

    # 4. Reshape back to N-dimensional grid if needed
    if grid:
        # jax.tree.map applies the reshape to all leaves (e.g. if eval_fn returns a tuple)
        reshape_fn = lambda x: x.reshape(out_shape_prefix + x.shape[1:]) if isinstance(x, jnp.ndarray) else x
        results = jax.tree.map(reshape_fn, results)

    return results