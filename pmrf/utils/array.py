"""
Utils for array manipulation and inspection.
"""

import jax.numpy as jnp
from jax import vmap
from pmrf.constants import Number

def find_nearest_index(array: jnp.ndarray, value: Number) -> int:
    """
    Find the nearest index for a value in array.

    Parameters
    ----------
    array :  np.ndarray
        array we are searching for a value in
    value : element of the array
        value to search for

    Returns
    --------
    found_index : int
        the index at which the  numerically closest element to `value`
        was found at

    References
    ----------
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    """
    return (jnp.abs(array-value)).argmin()


def slice_domain(x: jnp.ndarray, domain: tuple):
    """
    Returns a slice object closest to the `domain` of `x`

    domain = x[slice_domain(x, (start, stop))]

    Parameters
    ----------
    vector : np.ndarray
        an array of values
    domain : tuple
        tuple of (start,stop) values defining the domain over
        which to slice

    Examples
    --------
    >>> x = linspace(0,10,101)
    >>> idx = slice_domain(x, (2,6))
    >>> x[idx]

    """
    start = find_nearest_index(x, domain[0])
    stop = find_nearest_index(x, domain[1])
    return slice(start, stop+1)


def get_anomaly_mask(Y, threshold):
    """
    Computes a boolean mask of rows that ARE anomalies.
    Returns True for rows that exceed the threshold.
    """
    def check_row_is_anomaly(signal, th):
        d2 = jnp.diff(signal, n=2)
        curvature = jnp.abs(d2)
    
        th_is_vector = jnp.ndim(th) > 0
        threshold_view = th[1:-1] if th_is_vector else th
        has_spike = jnp.any(curvature > threshold_view)
        return has_spike

    mask = vmap(check_row_is_anomaly, in_axes=(0, None))(Y, threshold)
    return mask

def has_sudden_changes(values, *, upwards=True, downwards=True, window=5, iqr_factor=1.5, relative_epsilon=0.05):
    """
    Checks if an array has any sudden changes for all values relative to the previous `window` values.
    """
    values = jnp.array(values)

    for idx in range(len(values) - 1, window - 1, -1):
        # First detect any spikes
        target_value = values[idx]
        history_start = idx - window
        history_end = idx
        history_window = values[history_start:history_end]
        
        # Calculate IQR on that history
        Q1, Q3 = jnp.percentile(jnp.array(history_window), 25), jnp.percentile(jnp.array(history_window), 75)
        actual_iqr = Q3 - Q1
        min_iqr_floor = (jnp.abs(jnp.median(jnp.array(history_window))) * relative_epsilon)
        effective_iqr = jnp.maximum(actual_iqr, min_iqr_floor)
        
        if upwards:
            upper_threshold = Q3 + effective_iqr * iqr_factor
            if target_value > upper_threshold:
                return True
        if downwards:
            lower_threshold = Q1 - effective_iqr * iqr_factor
            if target_value < lower_threshold:
                return True
        
    return False


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