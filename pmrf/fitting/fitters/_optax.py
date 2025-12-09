# pmrf/fitting/_optax.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import jax
import jax.numpy as jnp
import optax

from pmrf.fitting._frequentist import FrequentistFitter, FrequentistResults

class OptaxFitter(FrequentistFitter):
    """
    JAX/Optax-based fitter. Uses gradient-based optimization on a flat
    parameter vector with box constraints handled by projection.
    """
    def _make_optimizer(
        self,
        name: str,
        lr: float,
        grad_clip_norm: Optional[float],
    ) -> optax.GradientTransformation:
        chain: List[optax.GradientTransformation] = []
        if grad_clip_norm is not None:
            chain.append(optax.clip_by_global_norm(grad_clip_norm))

        name_l = name.lower()
        if name_l == "adam":
            chain.append(optax.adam(lr))
        elif name_l == "adamw":
            chain.append(optax.adamw(lr))
        elif name_l == "rmsprop":
            chain.append(optax.rmsprop(lr))
        elif name_l == "sgd":
            chain.append(optax.sgd(lr))
        elif name_l == "lbfgs":
            chain.append(optax.lbfgs(lr))
        else:
            raise ValueError(f"Unknown optimizer '{name}'; use one of: adam, adamw, rmsprop, sgd")

        return optax.chain(*chain)    

    def run(
        self,
        *,
        optimizer: str = "adam",
        learning_rate: float = 1e-2,
        max_steps: int = 20_000,
        log_every: int = 500,
        grad_clip_norm: Optional[float] = None,
        plateau_patience: int = 2_000,
        plateau_tol: float = 1e-6,
        grad_tol: Optional[float] = 1e-6,
        seed: int = 0,
        **kwargs,
    ) -> FrequentistResults:
        """
        Parameters
        ----------
        optimizer:
            One of {"adam", "adamw", "rmsprop", "sgd", "lbfgs"}.
        learning_rate:
            Base learning rate for the optimizer.
        max_steps:
            Hard cap on total optimization steps.
        log_every:
            Print progress every N steps.
        grad_clip_norm:
            If provided, clip gradient global norm to this value.
        plateau_patience:
            Stop if best cost hasn't improved by >= plateau_tol over the last `plateau_patience` steps.
        plateau_tol:
            Absolute improvement threshold used for plateau detection.
        grad_tol:
            If provided, stop when ||grad||_2 <= grad_tol.
        seed:
            For any stochastic components (kept for future compatibility).
        kwargs:
            Reserved for compatibility; ignored here (you can plumb extra knobs if needed).
        """
        x0 = jnp.asarray(self.initial_model.flat_params(), dtype=jnp.float64)
        param_names = self.initial_model.flat_param_names()
        mins, maxs = self._bounds()
        mins = jnp.asarray(mins, dtype=jnp.float64)
        maxs = jnp.asarray(maxs, dtype=jnp.float64)

        if jnp.any((maxs - x0) < 0.0) or jnp.any((x0 - mins) < 0.0):
            raise Exception("Bad prior bounds")

        cost_fn = self._make_cost_function()
        opt = self._make_optimizer(optimizer, learning_rate, grad_clip_norm)
        value_and_grad = jax.jit(jax.value_and_grad(cost_fn))

        @jax.jit
        def project(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.clip(x, mins, maxs)

        @jax.jit
        def step(x, opt_state):
            val, g = value_and_grad(x)
            updates, opt_state = opt.update(g, opt_state, params=x)
            x_new = optax.apply_updates(x, updates)
            x_new = project(x_new)
            grad_norm = jnp.linalg.norm(g)
            return x_new, opt_state, val, grad_norm

        # --- Initialize
        x = project(x0)
        opt_state = opt.init(x)
        best_val = jnp.inf
        best_x = x
        steps_since_improve = 0

        # Logging
        self.logger.info(
            f"Fitting {len(x0)} parameters with optax-{optimizer} "
            f"(lr={learning_rate}, max_steps={max_steps})"
        )
        self.logger.info(f"Parameter names: {param_names}")

        # --- Main loop
        history_cost: List[float] = []
        history_grad_norm: List[float] = []

        for step_idx in range(1, max_steps + 1):
            x, opt_state, val, grad_norm = step(x, opt_state)
            # sync to host scalars for control flow/logging
            val_h = float(val)
            gnorm_h = float(grad_norm)

            history_cost.append(val_h)
            history_grad_norm.append(gnorm_h)

            # Track best
            if val_h + plateau_tol < float(best_val):
                best_val = val_h
                best_x = x
                steps_since_improve = 0
            else:
                steps_since_improve += 1

            # Periodic logs
            if (step_idx % log_every) == 0:
                self.logger.info(f"step = {step_idx}, cost = {val_h:.6g}, ||g|| = {gnorm_h:.3e}")

            # Early stopping: gradient norm
            if (grad_tol is not None) and (gnorm_h <= grad_tol):
                self.logger.info(
                    f"Stopping: gradient norm {gnorm_h:.3e} <= grad_tol {grad_tol:.3e} at step {step_idx}"
                )
                break

            # Early stopping: plateau
            if steps_since_improve >= plateau_patience:
                self.logger.info(
                    f"Stopping: no improvement >= {plateau_tol:g} for {plateau_patience} steps "
                    f"(best={float(best_val):.6g}, current={val_h:.6g})"
                )
                break

        final_x = best_x  # use best parameters encountered
        final_cost = float(best_val)

        fitted_model = self.initial_model.with_flat_params(jnp.asarray(final_x))
        settings = self._settings(
            dict(
                optimizer=optimizer,
                learning_rate=learning_rate,
                max_steps=max_steps,
                log_every=log_every,
                grad_clip_norm=grad_clip_norm,
                plateau_patience=plateau_patience,
                plateau_tol=plateau_tol,
                grad_tol=grad_tol,
                seed=seed,
                **kwargs,
            )
        )

        solver_results = dict(
            message="Optimization finished",
            status=0,  # 0=OK (mimic SciPy style)
            nit=len(history_cost),
            fun=final_cost,
            x=jnp.asarray(final_x),
            cost_history=jnp.asarray(history_cost),
            grad_norm_history=jnp.asarray(history_grad_norm),
        )

        self.logger.info(f"Finished optax-{optimizer}: steps={solver_results['nit']}, "f"best_cost={final_cost:.6g}")

        self.results = FrequentistResults(
            measured=self.measured,
            initial_model=self.initial_model,
            fitted_model=fitted_model,
            solver_results=solver_results,
            settings=settings,
        )

        return self.results