import logging
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize as scipy_minimize
from tqdm.auto import tqdm

DEBUG_PRINT = False

class ScipyMinimizerState(eqx.Module):
    """Holds the solver state across the single-step JAX loop."""
    step_count: jax.Array
    y_opt: Any  
    num_steps: jax.Array
    num_evals: jax.Array
    loss: jax.Array
    success: jax.Array


class ScipyMinimizer(optx.AbstractMinimiser):
    """
    A JAX-wrapped optimizer using `scipy.optimize.minimize`, compatible 
    with the optimistix.AbstractMinimiser API.
    """
    method: str = eqx.field(static=True, default="L-BFGS-B")
    options: dict = eqx.field(static=True, default_factory=dict)
    use_grad: bool = eqx.field(static=True, default=True)
    show_progress: bool = eqx.field(static=True, default=True)

    rtol: float = eqx.field(static=True, default=1e-5)
    atol: float = eqx.field(static=True, default=1e-5)
    norm: Callable = eqx.field(static=True, default=optx.max_norm)

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        """Initializes the mock solver state."""
        return ScipyMinimizerState(
            step_count=jnp.array(0, dtype=jnp.int32),
            y_opt=y,
            num_steps=jnp.array(0, dtype=jnp.int32),
            num_evals=jnp.array(0, dtype=jnp.int32),
            loss=jnp.array(jnp.inf, dtype=jnp.float64),
            success=jnp.array(False, dtype=jnp.bool_)
        )

    def step(self, fn, y, args, options, state, tags):
        """Performs the full SciPy optimization natively via a host callback."""
        method = self.method
        scipy_options = dict(self.options)
        lower_tree = options.get('lower', None)
        upper_tree = options.get('upper', None)

        gradient_free_methods = {'nelder-mead', 'powell', 'cobyla'}
        use_grad = self.use_grad and (method.lower() not in gradient_free_methods)

        flat_y, unravel_fn = ravel_pytree(y)

        has_bounds = lower_tree is not None and upper_tree is not None
        if has_bounds:
            flat_lower, _ = ravel_pytree(lower_tree)
            flat_upper, _ = ravel_pytree(upper_tree)
        else:
            flat_lower = jnp.array([], dtype=flat_y.dtype)
            flat_upper = jnp.array([], dtype=flat_y.dtype)

        @jax.jit
        def flat_fn(_flat_y, _args):
            out = fn(unravel_fn(_flat_y), _args)
            return out[0] if isinstance(out, tuple) else out

        val_and_grad_fn = jax.value_and_grad(flat_fn)

        def host_scipy_call(flat_y_np, flat_lower_np, flat_upper_np, args_np):
            scipy_bounds = None
            if has_bounds:
                scipy_bounds = list(zip(flat_lower_np, flat_upper_np))

            current_loss = [np.inf]

            def objective_with_grad(x_np):
                nan_logged = False
                loss, grad = val_and_grad_fn(jnp.array(x_np), args_np)
                loss_np = np.asarray(loss, dtype=np.float64)
                grad_np = np.asarray(grad, dtype=np.float64)
                current_loss[0] = loss_np
                
                if not nan_logged and np.any(np.isnan(loss_np)):
                    logging.warning("Loss value was NaN")
                    nan_logged = True
                    
                if not nan_logged and np.any(np.isnan(grad_np)):
                    logging.warning("Loss gradients were NaN")
                    nan_logged = True
                    
                return loss_np, grad_np

            def objective_no_grad(x_np):
                loss = flat_fn(jnp.array(x_np), args_np)
                loss_np = np.asarray(loss, dtype=np.float64)
                current_loss[0] = loss_np
                return loss_np

            obj_func = objective_with_grad if use_grad else objective_no_grad

            pbar = None
            if self.show_progress:
                maxiter = scipy_options.get("maxiter", None)
                pbar = tqdm(total=maxiter, desc=f"SciPy {method}")

            def callback(*cb_args, **cb_kwargs):
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{current_loss[0]:.3g}")

            try:
                res = scipy_minimize(
                    obj_func, 
                    np.array(flat_y_np), 
                    jac=use_grad, 
                    method=method,
                    bounds=scipy_bounds,  
                    options=scipy_options,
                    callback=callback,
                )
            finally:
                if pbar is not None:
                    pbar.close()

            return (
                res.x.astype(flat_y_np.dtype),
                np.int32(res.nit),
                np.int32(res.nfev),
                np.float64(res.fun),
                np.bool_(res.success)
            )

        result_shape_dtypes = (
            jax.ShapeDtypeStruct(flat_y.shape, flat_y.dtype),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.float64),
            jax.ShapeDtypeStruct((), jnp.bool_)
        )

        def run_optimization(_):
            return jax.pure_callback(
                host_scipy_call, 
                result_shape_dtypes, 
                flat_y, flat_lower, flat_upper, args
            )

        def skip_optimization(_):
            flat_y_opt, _ = ravel_pytree(state.y_opt)
            return flat_y_opt, state.num_steps, state.num_evals, state.loss, state.success

        y_opt_flat, nit, nfev, fun, success = jax.lax.cond(
            state.step_count == 0,
            run_optimization,
            skip_optimization,
            None
        )

        new_y = unravel_fn(y_opt_flat)
        new_state = ScipyMinimizerState(
            step_count=state.step_count + 1,
            y_opt=new_y,
            num_steps=nit,
            num_evals=nfev,
            loss=fun,
            success=success
        )

        out = fn(new_y, args)
        aux = out[1] if isinstance(out, tuple) else None

        return new_y, new_state, aux

    def terminate(self, fn, y, args, options, state, tags):
        """Evaluates whether to terminate or continue."""
        # Signal that the solver has finished stepping once step_count > 0
        done = state.step_count > 0
        
        # SciPy success flag to dictate the final optimistix state.
        is_ok = (state.step_count == 0) | state.success

        result = jax.lax.cond(
            is_ok,
            lambda _: optx.RESULTS.successful,
            lambda _: optx.RESULTS.max_steps_reached,
            None
        )
        
        return done, result

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        """Pushes performance metrics down to the caller."""
        stats = {
            "num_steps": state.num_steps,
            "num_evals": state.num_evals,
            "loss": state.loss
        }
        return y, aux, stats