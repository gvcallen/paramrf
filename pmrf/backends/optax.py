import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from pmrf.models.model import Model

def run_optax(
    model: Model,
    cost_fn: callable,
    logger,
    *,
    optimizer=None,
    max_iter=1000,
    learning_rate=1e-2,
    show_progress=True,
    atol=1e-5,
    patience=50,
    debug_save_loss=False,
    loss_file_path="optax_loss_log.csv",
    **kwargs
) -> tuple[Model, dict]:
    """
    Shared backend function for executing Optax optimization with JAX.
    Enforces box constraints via projected gradient descent.
    """
    # 1. Parameter Initialization & Bounds
    # Support both .bounds tuple and .min/.max attributes for compatibility
    dist = model.distribution()
    if hasattr(dist, 'bounds'):
        minimums, maximums = dist.bounds
    else:
        minimums, maximums = dist.min, dist.max
        
    min_bounds, max_bounds = jnp.array(minimums), jnp.array(maximums)
    x0 = jnp.array(model.flat_param_values())

    # Validate initial guess against bounds
    too_low, too_high = x0 < min_bounds, x0 > max_bounds
    if jnp.any(too_low | too_high):
        param_names = model.flat_param_names()
        bad_params = [
            f"  {name}: x0={val}, min={minv}, max={maxv} ({'below min' if low else 'above max'})"
            for name, val, minv, maxv, low, high in zip(param_names, x0, min_bounds, max_bounds, too_low, too_high)
            if low or high
        ]
        raise ValueError(f"Initial parameters outside bounds:\n" + "\n".join(bad_params))

    # 2. Setup Optimizer
    if optimizer is None or optimizer == 'adam':
        tx = optax.adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        tx = optax.sgd(learning_rate=learning_rate)
    elif isinstance(optimizer, optax.GradientTransformation):
        tx = optimizer
    else:
        raise ValueError("Optimizer must be 'adam', 'sgd', or an optax.GradientTransformation.")

    opt_state = tx.init(x0)

    # 3. Define the JAX-native step function
    loss_and_grad_fn = jax.value_and_grad(cost_fn)

    @jax.jit
    def step_fn(params, state):
        # Calculate loss and gradients automatically
        loss, grads = loss_and_grad_fn(params)
        
        # Apply Optax transformations
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        
        # Box Constraints: Project back into valid bounds
        params = jnp.clip(params, min_bounds, max_bounds)
        
        return params, state, loss

    # WARMUP: Execute once to trigger JAX compilation so we don't time it
    _ = step_fn(x0, opt_state)

    logger.info("Starting Optax optimization...")
    
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
            
            # Fetch concrete loss value for early stopping (syncs JAX)
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
    
    optimized_model = model.with_params(np.array(params))
    return optimized_model, optax_result