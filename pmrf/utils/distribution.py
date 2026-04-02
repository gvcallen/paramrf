import jax.numpy as jnp
import distreqx.distributions as dist

def unwrap_base_distribution(distribution: dist.AbstractDistribution, data: jnp.ndarray) -> tuple[dist.AbstractDistribution, jnp.ndarray, jnp.ndarray]:
    """
    Recursively unwraps nested `distreqx.Transformed` distributions.
    
    Returns the foundational base distribution, the data mapped into that 
    base coordinate space, and the accumulated log-determinant of the Jacobian.
    """
    base_dist = distribution
    base_data = data
    log_det_jacobian = 0.0

    while isinstance(base_dist, dist.Transformed):
        bijector = base_dist.bijector
        base_data, layer_log_det = bijector.inverse_and_log_det(base_data)
        log_det_jacobian += layer_log_det
        base_dist = base_dist.distribution
        
    return base_dist, base_data, log_det_jacobian