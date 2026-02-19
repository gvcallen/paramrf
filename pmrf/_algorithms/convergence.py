import jax.numpy as jnp

from pmrf._algorithms import has_sudden_changes

def has_converged_by_threshold(values, threshold):
    """
    Check if an array has converged by reaching below a threshold.
    """
    values = jnp.array(values)
    return values[-1] > threshold

def has_converged_by_patience(values, patience):
    """
    Check if an array has converged by no improvement over a recent window.
    """
    values = jnp.array(values)

    if len(values) < 2 * patience + 1:
        return False
    
    # We don't allow patience convergance if there are any recent sudden changes
    spike_start_idx = len(values) - (2 * patience + 1)
    if has_sudden_changes(values[spike_start_idx:], window=patience, downwards=False):
        return False

    # Check if our value is better than the best in window.
    current_window = values[-patience:]
    overall_best_so_far = jnp.min(values[:-patience])
    window_best = jnp.min(current_window)
    if window_best < overall_best_so_far:
        return False
    
    return True

def has_converged(values, *, threshold=None, patience=None, iqr_factor=1.5, relative_epsilon=0.05) -> bool:
    """
    Check if an array has converged by various metrics.
    """
    values = jnp.array(values)

    if threshold is not None and not has_converged_by_threshold(values, threshold):
        return False
    
    if patience is not None and not has_converged_by_patience(values, patience, iqr_factor=iqr_factor, relative_epsilon=relative_epsilon):
        return False
    
    return True