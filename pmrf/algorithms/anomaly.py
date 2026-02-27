import jax.numpy as jnp

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
