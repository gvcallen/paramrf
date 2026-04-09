import jax.numpy as jnp
from pmrf.utils.anomaly import has_sudden_changes

def has_converged_by_absolute_tolerance(values, atol, window=5):
    """
    Returns True if the last `window` values are all below the absolute tolerance.
    
    Parameters
    ----------
    values : array-like
        Sequence of convergence metric values (ordered in time).
    atol : float
        Absolute tolerance.
    window : int
        Number of consecutive values required for convergence.
    """
    values = jnp.array(values)

    if len(values) < window:
        return False

    recent = values[-window:]

    return jnp.all(recent < atol)

def has_converged_by_relative_tolerance(values, rtol=0.01, window=5):
    """
    Returns True if the relative change between consecutive values
    is below epsilon for the last `window` steps.
    """
    values = jnp.asarray(values)

    if len(values) < window + 1:
        return False

    recent = values[-(window + 1):]
    rel_changes = jnp.abs(jnp.diff(recent)) / jnp.abs(recent[:-1])

    return jnp.all(rel_changes < rtol)

def has_converged_by_patience(values, patience, iqr_factor=1.5, relative_epsilon=0.05):
    """
    Check if an array has converged by no improvement over a recent window.
    """
    values = jnp.array(values)

    if len(values) < 2 * patience + 1:
        return False
    
    # We don't allow patience convergance if there are any recent sudden changes
    spike_start_idx = len(values) - (2 * patience + 1)
    if has_sudden_changes(values[spike_start_idx:], window=patience, downwards=False, iqr_factor=iqr_factor, relative_epsilon=relative_epsilon):
        return False

    # Check if our value is better than the best in window.
    current_window = values[-patience:]
    overall_best_so_far = jnp.min(values[:-patience])
    window_best = jnp.min(current_window)
    if window_best < overall_best_so_far:
        return False
    
    return True

def has_converged(values, *, rtol=None, atol=None, patience=None, window=5, iqr_factor=1.5, relative_epsilon=0.05) -> bool:
    """
    Check if an array has converged by various metrics.
    """
    values = jnp.array(values)

    if rtol is not None and not has_converged_by_relative_tolerance(values, rtol, window=window):
        return False
    
    if atol is not None and not has_converged_by_absolute_tolerance(values, atol, window=window):
        return False
    
    if patience is not None and not has_converged_by_patience(values, patience, iqr_factor=iqr_factor, relative_epsilon=relative_epsilon):
        return False
    
    return True