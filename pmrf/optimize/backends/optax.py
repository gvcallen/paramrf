import time
import logging
from typing import Callable, Sequence, Any
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from tqdm.auto import tqdm

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.optimize.problem import FrequentistProblem

def optimize_optax(
    model: Model,
    cost: Callable[[Model, Frequency], jnp.ndarray | float] | Sequence,
    frequency: Frequency,
    *,
    logger: logging.Logger | None = None,
    optimizer: str | optax.GradientTransformation = 'adam',
    max_iter: int = 1000,
    learning_rate: float = 1e-2,
    show_progress: bool = True,
    atol: float = 1e-5,
    patience: int = 50,
    debug_save_loss: bool = False,
    loss_file_path: str = "optax_loss_log.csv",
    **kwargs
) -> tuple[Model, dict[str, Any]]:
    """
    Executes Optax optimization with JAX acceleration.
    Enforces box constraints via projected gradient descent.
    """
    logger = logger or logging.getLogger(__name__)

    # 1. Setup the pure functional problem 
    problem = FrequentistProblem(model, cost, frequency)
    
    x0 = problem.x0
    min_bounds, max_bounds = problem.bounds

    # 2. Setup Optimizer
    if optimizer == 'adam':
        tx = optax.adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        tx = optax.sgd(learning_rate=learning_rate)
    elif isinstance(optimizer, optax.GradientTransformation):
        tx = optimizer
    else:
        raise ValueError("Optimizer must be 'adam', 'sgd', or an optax.GradientTransformation.")

    opt_state = tx.init(x0)

    # 3. Define the JAX-native step function
    loss_and_grad_fn = jax.value_and_grad(problem.flat_cost_fn)

    
    @eqx.filter_jit
    def step_fn(params, state):
        # Calculate loss and exact gradients automatically
        loss, grads = loss_and_grad_fn(params)
        
        # Apply Optax transformations (momentum, Adam scaling, etc.)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        
        # Box Constraints: Project back into valid physical bounds
        params = jnp.clip(params, min_bounds, max_bounds)
        
        
        return params, state, loss

    # WARMUP: Execute once to trigger XLA compilation
    logger.debug("JIT compiling Optax step function...")
    _ = step_fn(x0, opt_state)

    logger.info(f"Starting Optax optimization ({optimizer})...")
    
    # Tracking setup
    loss_history = []
    time_history = []
    
    # 4. Optimization Loop with Early Stopping
    params = x0
    current_loss = float('inf')
    best_loss = float('inf')
    patience_counter = 0
    actual_steps = 0
    stop_reason = "Maximum iterations reached."
    
    # Start the clock after warmup
    t0 = time.perf_counter()
    
    with tqdm(total=max_iter, desc="Optimizing", unit=" step", disable=not show_progress) as pbar:
        for i in range(max_iter):
            params, opt_state, loss = step_fn(params, opt_state)
            actual_steps += 1
            
            # Fetch concrete loss value for early stopping (syncs JAX with Python)
            current_loss = float(loss)
            
            if debug_save_loss:
                loss_history.append(current_loss)
                time_history.append(time.perf_counter() - t0)
            
            # Check early stopping criteria
            if current_loss < best_loss - atol:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update progress bar
            pbar.set_postfix({'cost': f"{current_loss:.4f}", 'patience': f"{patience_counter}/{patience}"})
            pbar.update(1)
            
            # Break if patience is exceeded
            if patience_counter >= patience:
                stop_reason = f"Early stopping triggered at step {i} (patience={patience} exhausted)."
                break

    logger.info(f"Optimization finished. Cost: {current_loss:.4f}, Steps: {actual_steps}. Reason: {stop_reason}")

    # Save tracking history
    if debug_save_loss and loss_history:
        try:
            data_to_save = np.column_stack((time_history, loss_history))
            np.savetxt(loss_file_path, data_to_save, header="time_seconds,loss", comments="", delimiter=",")
        except Exception as e:
            logger.error(f"Failed to save Optax loss history. Error: {e}")

    # 5. Package and Return
    optax_result = {
        'x': np.array(params),
        'fun': current_loss,
        'success': patience_counter < patience or actual_steps == max_iter,
        'message': stop_reason,
        'nfev': actual_steps,
        'nit': actual_steps
    }
    
    # Safely reconstruct the full PyTree model
    optimized_model = problem.reconstruct_fn(params)
    return optimized_model, optax_result