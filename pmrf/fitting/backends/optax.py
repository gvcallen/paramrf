import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm.auto import tqdm

from pmrf.fitting.frequentist import FrequentistFitter
from pmrf.models.model import Model

class OptaxFitter(FrequentistFitter):
    """
    Frequentist fitter using the Optax and JAX backend.
    
    This class leverages JAX's automatic differentiation and Optax's gradient 
    transformations to optimize the model parameters. Box constraints are 
    enforced via projected gradient descent (clipping parameters after each step).
    """
    
    def execute(
        self, 
        target: jnp.ndarray, 
        *, 
        optimizer=None, 
        max_iter=10000, 
        learning_rate=1e-3,
        show_progress=True,
        atol=1e-5,       # New: Minimum loss improvement to reset patience
        patience=50,    # New: Number of steps to wait without improvement
    ) -> tuple[Model, dict]:
        """
        Run the optimization loop using the Optax backend.
        
        **NB:** This method should not be called directly. Call `run` instead.

        Parameters
        ----------
        target : jax.numpy.ndarray
            The extracted target features to fit against.
        optimizer : str or optax.GradientTransformation, optional
            The optimizer to use. Basic users can pass 'adam' or 'sgd' (defaults to 'adam'). 
        max_iter : int, default=10000
            The maximum number of optimization steps to perform.
        learning_rate : float, default=1e-2
            The learning rate for the default optimizer.
        show_progress : bool, default=True
            Whether to display a `tqdm` progress bar tracking the loss.
        atol : float, default=1e-5
            The minimum required improvement in the loss to reset the early stopping counter.
        patience : int, default=50
            The number of steps to continue optimizing without an improvement >= `atol` 
            before triggering early stopping.

        Returns
        -------
        tuple[:class:`~pmrf.models.model.Model`, dict]
            The fitted model and a mock SciPy result object containing the final state.
        """
        # 1. Parameter Initialization & Bounds
        minimums, maximums = self.model.distribution().bounds
        min_bounds, max_bounds = jnp.array(minimums), jnp.array(maximums)
        
        x0 = jnp.array(self.model.flat_param_values())

        # Validate initial guess against bounds
        too_low, too_high = x0 < min_bounds, x0 > max_bounds
        if jnp.any(too_low | too_high):
            param_names = self.model.flat_param_names()
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
        def compute_loss(params):
            return self.cost(params, target)
            
        loss_and_grad_fn = jax.value_and_grad(compute_loss)

        def step_fn(params, state):
            # Calculate loss and gradients automatically
            loss, grads = loss_and_grad_fn(params)
            
            # Apply Optax transformations
            updates, state = tx.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            
            # Box Constraints: Project back into valid bounds
            params = jnp.clip(params, min_bounds, max_bounds)
            
            return params, state, loss

        step_fn = jax.jit(step_fn)

        self.logger.info("Starting Optax optimization...")
        
        # 4. Optimization Loop with Early Stopping
        params = x0
        current_loss = float('inf')
        
        # Early stopping trackers
        best_loss = float('inf')
        patience_counter = 0
        actual_steps = 0
        stop_reason = "Maximum iterations reached."
        
        with tqdm(total=max_iter, desc="Optimizing", unit=" step", disable=not show_progress) as pbar:
            for i in range(max_iter):
                params, opt_state, loss = step_fn(params, opt_state)
                actual_steps += 1
                
                # We need the concrete loss value for early stopping.
                # Note: fetching the loss to the host syncs JAX, which has a small overhead, 
                # but is necessary for Python-level early stopping.
                current_loss = float(loss)
                
                # Check early stopping criteria
                if current_loss < best_loss - atol:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Update progress bar
                if i % 10 == 0 or i == max_iter - 1:
                    pbar.set_postfix({'cost': f"{current_loss:.4f}", 'patience': f"{patience_counter}/{patience}"})
                pbar.update(1)
                
                # Break if patience is exceeded
                if patience_counter >= patience:
                    stop_reason = f"Early stopping triggered at step {i} (patience={patience} exhausted)."
                    break

        self.logger.info(f"Optimization finished. Cost: {current_loss:.4f}, Steps: {actual_steps}. Reason: {stop_reason}")

        # 5. Package and Return
        optax_result = {
            'x': np.array(params),
            'fun': current_loss,
            'success': True,
            'message': stop_reason,
            'nfev': actual_steps,
            'nit': actual_steps
        }
        
        fitted_model = self.model.with_params(params)
        return fitted_model, optax_result