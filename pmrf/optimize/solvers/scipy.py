"""
SciPy optimization wrappers.
"""

from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree
import equinox as eqx
from tqdm.auto import tqdm
import numpy as np
from scipy.optimize import Bounds, minimize as scipy_minimize

from pmrf.optimize.base import AbstractBoundedMinimizer, MinimizeResult

# The methods of :func:`scipy.optimize.minimize` that keep their iterates inside bounds.
_BOUNDED_METHODS = {'l-bfgs-b', 'tnc', 'slsqp', 'trust-constr', 'powell', 'nelder-mead', 'cobyla', 'cobyqa'}


class ScipyMinimize(AbstractBoundedMinimizer):
    """
    A wrapper around SciPy's :func:`scipy.optimize.minimize`.

    Whether it honours bounds depends on `method`: L-BFGS-B, TNC, SLSQP, trust-constr,
    Powell, Nelder-Mead, COBYLA and COBYQA do, and search box space; any other method,
    such as BFGS or CG, moves through raw space. The default, None, is L-BFGS-B, which
    SciPy picks when it is given bounds. When bounds are supplied to trust-constr,
    SciPy searches fixed affine coordinates centered at the box start and receives
    feasible bounds; `metrics.box_origin` and `metrics.box_scale` map its coordinates
    back to box space.

    Nonfinite objectives, requested gradients or attempted coordinates raise
    :class:`FloatingPointError` with the method and evaluation number before SciPy can
    return a failed start point. Finite nonconvergence keeps SciPy's usual result.
    """
    method: str | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)
    options: dict = eqx.field(static=True, default_factory=dict)
    show_progress: bool = eqx.field(static=True, default=True)
    use_grad: bool | None = eqx.field(static=True, default=None)
    # use_hess: bool | None = eqx.field(static=True, default=None)

    @property
    def honours_bounds(self) -> bool:
        """Whether `method` keeps its iterates inside bounds. None is L-BFGS-B."""
        return self.method is None or (isinstance(self.method, str) and self.method.lower() in _BOUNDED_METHODS)

    def run(
        self, 
        fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: Any = None,
        bounds: tuple[PyTree, PyTree] | None = None,
        max_iter: int = 1024,
        **kwargs
    ) -> MinimizeResult:
        options = self.options
        method = self.method
        tol = self.tol
        use_grad = self.use_grad

        if use_grad is None:
            gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
            if method is not None and (method.lower() in gradient_free_methods):
                use_grad = False
            else:
                use_grad = True

        if 'max_iter' in options:
            raise ValueError("Cannot pass `max_iter` in SciPy options")

        lower_tree, upper_tree = bounds if bounds is not None else (None, None)

        flat_y, unravel_fn = ravel_pytree(y0)
        scipy_options = dict(options)
        scipy_options.setdefault('maxiter', max_iter)

        scipy_bounds = None
        if lower_tree is not None and upper_tree is not None:
            flat_lower, _ = ravel_pytree(lower_tree)
            flat_upper, _ = ravel_pytree(upper_tree)
            flat_lower = np.asarray(flat_lower, dtype=np.float64)
            flat_upper = np.asarray(flat_upper, dtype=np.float64)
            scipy_bounds = list(zip(flat_lower, flat_upper))

        use_affine = (
            scipy_bounds is not None
            and isinstance(method, str)
            and method.lower() == 'trust-constr'
        )
        box_origin = np.asarray(flat_y, dtype=np.float64)
        box_scale = np.ones_like(box_origin)
        scipy_start = box_origin
        if use_affine:
            lower = np.asarray(flat_lower)
            upper = np.asarray(flat_upper)
            finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
            both = finite_lower & finite_upper
            box_scale = np.where(both, upper - lower, box_scale)
            box_scale = np.where(finite_lower & ~finite_upper, np.abs(box_origin - lower), box_scale)
            box_scale = np.where(~finite_lower & finite_upper, np.abs(upper - box_origin), box_scale)
            box_scale = np.where(~finite_lower & ~finite_upper, np.abs(box_origin), box_scale)
            box_scale = np.where(np.isfinite(box_scale) & (box_scale != 0), np.abs(box_scale), 1.0)
            scipy_bounds = Bounds(
                (lower - box_origin) / box_scale,
                (upper - box_origin) / box_scale,
                keep_feasible=True,
            )
            scipy_start = np.zeros_like(box_origin)
            method = method.lower()

        def flat_fn(_flat_y):
            return fn(unravel_fn(_flat_y), args)
            
        val_and_grad_fn = jax.jit(jax.value_and_grad(flat_fn))
        val_only_fn = jax.jit(flat_fn)

        current_loss = [np.inf]
        eval_count = [0]

        def coordinate_labels(tree):
            labels = []
            path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
            for path, leaf in path_leaves:
                parts = []
                for key in path:
                    if hasattr(key, 'key'):
                        parts.append(str(key.key))
                    elif hasattr(key, 'name'):
                        parts.append(str(key.name))
                    elif hasattr(key, 'idx'):
                        parts.append(str(key.idx))
                label = '.'.join(parts) if parts else None
                size = int(np.size(leaf))
                labels.extend([label] * size)
            return labels

        labels = coordinate_labels(y0)

        def label_list(indices):
            named = sorted({labels[i] for i in indices if i < len(labels) and labels[i]})
            if named:
                return ', '.join(repr(name) for name in named)
            coordinates = ', '.join(str(int(i)) for i in indices)
            return f'coordinate indices {coordinates}'

        def original_coordinates(z):
            return box_origin + box_scale * z if use_affine else z

        def checked_input(x_np):
            eval_count[0] += 1
            x_np = np.asarray(x_np, dtype=np.float64)
            nonfinite = np.flatnonzero(~np.isfinite(x_np))
            if len(nonfinite):
                raise FloatingPointError(
                    f"SciPy {method or 'default'} objective evaluation {eval_count[0]} "
                    "received a nonfinite attempted optimizer vector for "
                    f"{label_list(nonfinite)}."
                )
            return original_coordinates(x_np)

        def checked_loss(loss):
            loss_float = float(np.asarray(loss))
            if not np.isfinite(loss_float):
                raise FloatingPointError(
                    f"SciPy {method or 'default'} objective evaluation {eval_count[0]} "
                    "produced a nonfinite loss; free parameters in context: "
                    f"{label_list(range(len(labels)))}."
                )
            return loss_float

        def objective_with_grad(x_np):
            box_x = checked_input(x_np)
            loss, grad = val_and_grad_fn(jnp.asarray(box_x))
            loss_float = checked_loss(loss)
            grad_np = np.asarray(grad, dtype=np.float64)
            if use_affine:
                grad_np = grad_np * box_scale
            bad_grad = np.flatnonzero(~np.isfinite(grad_np))
            if len(bad_grad):
                raise FloatingPointError(
                    f"SciPy {method or 'default'} objective evaluation {eval_count[0]} "
                    f"produced a nonfinite gradient for {label_list(bad_grad)}."
                )
            current_loss[0] = loss_float
            return loss_float, grad_np

        def objective_no_grad(x_np):
            box_x = checked_input(x_np)
            loss = val_only_fn(jnp.asarray(box_x))
            loss_float = checked_loss(loss)
            current_loss[0] = loss_float
            return loss_float

        obj_func = objective_with_grad if use_grad else objective_no_grad

        pbar = None
        if self.show_progress:
            maxiter = scipy_options.get("maxiter", None)
            desc = f"SciPy {method}" if method is not None else "SciPy (default)"
            pbar = tqdm(total=maxiter, desc=desc)

        def callback(*cb_args, **cb_kwargs):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{current_loss[0]:.3g}")

        try:
            res = scipy_minimize(
                obj_func, 
                scipy_start,
                jac=use_grad, 
                # hess=hess_arg,
                method=method,
                tol=tol,
                bounds=scipy_bounds,  
                options=scipy_options,
                callback=callback,
                **kwargs,
            )
        finally:
            if pbar is not None:
                pbar.close()

        result_x = np.asarray(res.x, dtype=np.float64)
        box_x = box_origin + box_scale * result_x if use_affine else result_x
        if use_affine:
            res.box_origin = box_origin.copy()
            res.box_scale = box_scale.copy()
        return MinimizeResult(
                y=unravel_fn(jnp.asarray(box_x)),
            success=bool(res.success), 
            metrics=res
        )
