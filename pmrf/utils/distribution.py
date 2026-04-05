import jax
import jax.numpy as jnp
import distreqx.distributions as dist
import distreqx.bijectors as bij
from jaxtyping import Array


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

def _batched_mvn(loc: Array, cov: Array) -> dist.MultivariateNormalFullCovariance:
    """Helper to cleanly initialize batched MVNs."""
    batch_ndims = loc.ndim - 1
    init_fn = dist.MultivariateNormalFullCovariance
    for _ in range(batch_ndims):
        init_fn = jax.vmap(init_fn)
    return init_fn(loc, cov)

def unwrap_base_distribution(distribution: dist.AbstractDistribution, data: jnp.ndarray) -> tuple[dist.AbstractDistribution, jnp.ndarray, jnp.ndarray]:
    """Recursively unwraps nested `distreqx.Transformed` distributions."""
    base_dist = distribution
    base_data = data
    log_det_jacobian = 0.0

    while isinstance(base_dist, dist.Transformed):
        bijector = base_dist.bijector
        base_data, layer_log_det = bijector.inverse_and_log_det(base_data)
        log_det_jacobian += layer_log_det
        base_dist = base_dist.distribution
        
    return base_dist, base_data, log_det_jacobian

def make_complex_normal(
    loc: Array, 
    covariance: Array, 
    pseudo_covariance: Array | None = None
) -> dist.Transformed:
    """
    Constructs a transformed complex Gaussian distribution from complex parameters.
    """
    loc_r2 = jnp.stack([jnp.real(loc), jnp.imag(loc)], axis=-1)
    
    if pseudo_covariance is None:
        variance_r2 = jnp.real(covariance) / 2.0
        scale_r2 = jnp.sqrt(variance_r2)
        scale_r2 = jnp.expand_dims(scale_r2, axis=-1)
        
        # distreqx Independent just takes the distribution
        base_dist = dist.Independent(dist.Normal(loc=loc_r2, scale=scale_r2))
    else:
        gamma = jnp.real(covariance) 
        c_real = jnp.real(pseudo_covariance)
        c_imag = jnp.imag(pseudo_covariance)
        
        cov_11 = 0.5 * (gamma + c_real)
        cov_22 = 0.5 * (gamma - c_real)
        cov_12 = 0.5 * c_imag
        
        row1 = jnp.stack([cov_11, cov_12], axis=-1)
        row2 = jnp.stack([cov_12, cov_22], axis=-1)
        covariance_matrix_r2 = jnp.stack([row1, row2], axis=-2)
        
        # Safe batched constructor
        base_dist = _batched_mvn(loc_r2, covariance_matrix_r2)
    
    return dist.Transformed(distribution=base_dist, bijector=bij.R2ToComplex())