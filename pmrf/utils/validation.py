import jax.numpy as jnp

def validate_bounds(x0, minimums, maximums, param_names):
    too_low, too_high = x0 < minimums, x0 > maximums
    if jnp.any(too_low | too_high):
        bad_params = [
            f"  {name}: x0={val}, min={minv}, max={maxv} ({'below min' if low else 'above max'})"
            for name, val, minv, maxv, low, high in zip(param_names, x0, minimums, maximums, too_low, too_high)
            if low or high
        ]
        raise ValueError("Initial parameters outside bounds:\n" + "\n".join(bad_params))      