from typing import Callable, Any, Literal
import dataclasses

from jaxtyping import Array, PyTree
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import parax as prx
from parax._to_deprecate.shadowing import make_bounds_bijectors, apply_hypercube_transform
from parax.deprecate.extraction import extract_bounds, extract_distributions, extract_group_ids, extract_metadata

def parax_minimize(
    fn: Callable[[PyTree], Array],
    module: PyTree,
    solver: optx.AbstractMinimiser | Callable,
    search_space: Literal['latent', 'hypercube'] = 'latent',
    supports_bounded: bool = False,
    use_bounds: bool = True,
    icdf_bounds: float = 0.001,
    options: dict[str, Any] = None,
    max_steps: int | None = 1024,
    filter_spec: Any = None,
    **kwargs,
) -> tuple[PyTree, Any]:
    """
    Minimize a PyTree containing Parax parameters.
    """
    if search_space not in ('latent', 'hypercube'):
        raise ValueError(f"search_space must be 'latent' or 'hypercube', got '{search_space}'")
    
    options = options or {}
    if filter_spec is None:
        filter_spec = prx.where_free_param_raw_value(module)

    # -------------------------------------------------------------------------
    # 1. Extraction: Build parax-aware shadow trees
    # -------------------------------------------------------------------------
    def _get_meta(p, attr, default):
        if not prx.is_free_param(p): return None
        return getattr(p.metadata, attr, default) if p.metadata else default

    scale_tree = jax.tree.map(lambda p: _get_meta(p, 'scale', 1.0), module, is_leaf=prx.is_free_param)
    transform_tree = jax.tree.map(lambda p: _get_meta(p, 'transform', None), module, is_leaf=prx.is_free_param)
    
    dist_tree = extract_distributions(module)
    group_tree = extract_group_ids(module)

    # Extract bounds and auto-generate fallback bijectors if needed
    lower_tree, upper_tree = extract_bounds(module, icdf_bounds, 1.0 - icdf_bounds) if search_space == 'hypercube' else extract_bounds(module)
    bounds_bij_tree = make_bounds_bijectors(lower_tree, upper_tree)

    # Resolve active transforms: Explicit metadata.transform wins over generated bounds
    def _resolve_transform(t, b_t):
        if t is not None: return t
        return b_t if (use_bounds and not supports_bounded) else None
        
    active_transforms = jax.tree.map(_resolve_transform, transform_tree, bounds_bij_tree)

    # -------------------------------------------------------------------------
    # 2. Setup Optimizer State (Y Space)
    # -------------------------------------------------------------------------
    # Extract the raw latent_values to be optimized
    def _get_latent(p): return p.latent_value if prx.is_free_param(p) else p
    y_init_tree = jax.tree.map(_get_latent, module, is_leaf=prx.is_free_param)
    
    # Partition into active parameters and static context
    y_params, y_static = eqx.partition(y_init_tree, filter_spec=filter_spec)

    # If the solver supports bounds, supply them partitioned to match y_params
    if use_bounds and supports_bounded:
        lower_opt, _ = eqx.partition(lower_tree, filter_spec=filter_spec)
        upper_opt, _ = eqx.partition(upper_tree, filter_spec=filter_spec)
        options.setdefault('lower', lower_opt)
        options.setdefault('upper', upper_opt)

    # -------------------------------------------------------------------------
    # 3. Objective Function (Y -> X -> Theta -> fn)
    # -------------------------------------------------------------------------
    def obj_fn(y_params_opt, _args=None):
        # Reconstruct the full tree of Y values
        y_tree = eqx.combine(y_params_opt, y_static)
        
        # Map Y (Optimizer space) to X (Unscaled physical space)
        if search_space == 'hypercube':
            # Clip U to avoid infinities before mapping
            u_tree = jax.tree.map(lambda u: jnp.clip(u, icdf_bounds, 1.0 - icdf_bounds), y_tree)
            x_tree = apply_hypercube_transform(u_tree, dist_tree, group_tree)
        else:
            # Map Latent to Unscaled Physical using active transforms
            def _apply_t(y, t):
                if t is None: return y
                if hasattr(t, 'forward'): return t.forward(y) # distreqx Bijector
                return t(y) # Standard callable
            x_tree = jax.tree.map(_apply_t, y_tree, active_transforms)

        # Map X to Theta (Scaled physical space)
        def _apply_scale(x, s): return x * s if s is not None else x
        theta_tree = jax.tree.map(_apply_scale, x_tree, scale_tree)

        # Inject pure Theta arrays into a fully unwrapped module
        def _unwrap(p, theta):
            return theta if prx.is_free_param(p) else p
        unwrapped_module = jax.tree.map(_unwrap, module, theta_tree, is_leaf=prx.is_free_param)
        
        return fn(unwrapped_module)

    # -------------------------------------------------------------------------
    # 4. Run Optimization and Reconstruct
    # -------------------------------------------------------------------------
    if isinstance(solver, optx.AbstractMinimiser):
        solver_results = optx.minimise(obj_fn, solver, y_params, args=None, options=options, max_steps=max_steps, **kwargs)
    else:
        solver_results = solver(obj_fn, y_params, args=None, options=options, max_steps=max_steps, **kwargs)

    # Package the optimized latent variables back into the Module
    final_y_tree = eqx.combine(solver_results.value, y_static)
    
    def _repack(p, final_y):
        if prx.is_free_param(p):
            return dataclasses.replace(p, latent_value=final_y)
        return p
        
    optimized_mod = jax.tree.map(_repack, module, final_y_tree, is_leaf=prx.is_free_param)

    return optimized_mod, solver_results