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


def make_complex_normal(
    loc: Array, 
    covariance: Array, 
    pseudo_covariance: Array | None = None
) -> dist.Transformed:
    """
    Constructs a transformed complex Gaussian distribution from complex parameters.
    
    When `pseudo_covariance` is not provided, the distribution is assumed to be 
    circularly-symmetric. In this case, an optimized path is used, falling back 
    to an independent `distrax.Normal` instead of computing a full covariance matrix.

    Parameters
    ----------
    loc : Array
        Complex mean array of shape (...).
    covariance : Array
        Real-valued variance array of shape (...). 
        (Often called Hermitian covariance, though real for scalars).
    pseudo_covariance : Array, optional
        Complex pseudo-covariance array of shape (...). If None, the distribution 
        is assumed to be circularly-symmetric (where pseudo-covariance is 0), 
        and an optimized Independent normal distribution is used under the hood. 
        Default is None.

    Returns
    -------
    dist.Transformed
        A distrax.Transformed distribution over complex numbers.
    """
    # 1. Map the complex mean to an R2 vector: shape (..., 2)
    loc_r2 = jnp.stack([jnp.real(loc), jnp.imag(loc)], axis=-1)
    
    if pseudo_covariance is None:
        # --- OPTIMIZED PATH ---
        # Circularly-symmetric case: Real and imaginary parts are independent.
        # Var(Re) = Var(Im) = Gamma / 2
        variance_r2 = jnp.real(covariance) / 2.0
        scale_r2 = jnp.sqrt(variance_r2)
        
        # Expand dims to broadcast correctly against the (..., 2) loc array
        scale_r2 = jnp.expand_dims(scale_r2, axis=-1)
        
        base_dist = dist.Independent(
            dist.Normal(loc=loc_r2, scale=scale_r2),
            reinterpreted_batch_ndims=1
        )
    else:
        # --- FULL COVARIANCE PATH ---
        # General complex normal case
        gamma = jnp.real(covariance) 
        c_real = jnp.real(pseudo_covariance)
        c_imag = jnp.imag(pseudo_covariance)
        
        # Construct the 2x2 covariance matrix: shape (..., 2, 2)
        cov_11 = 0.5 * (gamma + c_real)
        cov_22 = 0.5 * (gamma - c_real)
        cov_12 = 0.5 * c_imag
        
        row1 = jnp.stack([cov_11, cov_12], axis=-1)
        row2 = jnp.stack([cov_12, cov_22], axis=-1)
        covariance_matrix_r2 = jnp.stack([row1, row2], axis=-2)
        
        base_dist = dist.MultivariateNormalFullCovariance(
            loc=loc_r2, 
            covariance_matrix=covariance_matrix_r2
        )
    
    # Apply the bijector to map R2 -> C
    return dist.Transformed(distribution=base_dist, bijector=bij.R2ToComplex())