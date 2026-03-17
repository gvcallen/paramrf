import logging
from typing import Any

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import optax
from tqdm.auto import tqdm

from optimize.base import AbstractOptimizer
from pmrf.optimize.problem import OptimizeProblem
from pmrf.optimize.result import OptimizeResult

class OptaxOptimizer(AbstractOptimizer):
    max_iter: int
    optimizer_algo: Any
    learning_rate: float
    atol: float
    patience: int
    show_progress: bool

    def __init__(
        self, 
        max_iter: int = 1000, 
        optimizer: str | optax.GradientTransformation = 'adam', 
        learning_rate: float = 1e-2, 
        atol: float = 1e-5, 
        patience: int = 50, 
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.atol = atol
        self.patience = patience

        # Resolve the optimizer choice
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                self.optimizer_algo = optax.adam(learning_rate=self.learning_rate)
            elif optimizer.lower() == 'sgd':
                self.optimizer_algo = optax.sgd(learning_rate=self.learning_rate)
            else:
                raise ValueError(f"Unsupported string optimizer: {optimizer}. Use 'adam' or 'sgd'.")
        elif isinstance(optimizer, optax.GradientTransformation):
            self.optimizer_algo = optimizer
        else:
            raise TypeError("optimizer must be a string ('adam', 'sgd') or an optax.GradientTransformation.")

    def solve(self, problem: OptimizeProblem, show_progress=True, **kwargs) -> OptimizeResult:
        logger = logging.getLogger(__name__)
        logger.info("Starting Optax optimization in unbounded space...")

        # 1. Setup the unbounded functional problem
        # We optimize y in [-inf, inf], so we NEVER need jnp.clip for bounds!
        unbounded_cost_fn = problem.make_unbounded_cost_fn()
        y0 = problem.flat_unbounded_initial_guess

        # 2. Initialize Optax State
        opt_state = self.optimizer_algo.init(y0)

        # 3. Define the JAX-native step function
        loss_and_grad_fn = eqx.filter_jit(jax.value_and_grad(unbounded_cost_fn))

        @eqx.filter_jit
        def step_fn(y, state):
            loss, grads = loss_and_grad_fn(y)
            updates, new_state = self.optimizer_algo.update(grads, state, y)
            new_y = optax.apply_updates(y, updates)
            return new_y, new_state, loss

        # WARMUP: Execute once to trigger XLA compilation
        logger.debug("JIT compiling Optax step function...")
        _ = step_fn(y0, opt_state)

        # 4. Optimization Loop with Early Stopping
        current_y = y0
        current_loss = float('inf')
        best_loss = float('inf')
        patience_counter = 0
        actual_steps = 0
        stop_reason = "Maximum iterations reached."

        with tqdm(total=self.max_iter, desc="Optimizing", unit=" step", disable=not self.show_progress) as pbar:
            for i in range(self.max_iter):
                current_y, opt_state, loss = step_fn(current_y, opt_state)
                actual_steps += 1
                
                # Fetch concrete loss value for early stopping
                current_loss = float(loss)
                
                # Check early stopping criteria
                if current_loss < best_loss - self.atol:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                pbar.set_postfix({'cost': f"{current_loss:.4f}", 'patience': f"{patience_counter}/{self.patience}"})
                pbar.update(1)
                
                # Break if patience is exceeded
                if patience_counter >= self.patience:
                    stop_reason = f"Early stopping triggered at step {i} (patience={self.patience} exhausted)."
                    break

        success = (patience_counter < self.patience) or (actual_steps == self.max_iter)
        logger.info(f"Optimization finished. Cost: {current_loss:.4f}, Steps: {actual_steps}. Reason: {stop_reason}")

        # 5. Map the winning unbounded array back to Physical Reality
        best_u = jnp.clip(jnn.sigmoid(current_y), 1e-7, 1.0 - 1e-7)
        best_physical_x = problem.model.distribution().icdf(best_u)
        optimized_model = problem.reconstruct_fn(best_physical_x)

        # 6. Package and Return
        return OptimizeResult(
            model=optimized_model,
            cost=current_loss,
            history={
                'message': stop_reason,
                'steps': actual_steps,
                'optax_state': opt_state
            },
            success=success
        )