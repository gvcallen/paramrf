import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
import distreqx.distributions as dist
import distreqx.bijectors as bij

import jax
import jax.numpy as jnp

def distribution_hypercube_to_physical(d: dist.AbstractDistribution, u: jnp.ndarray) -> jnp.ndarray:
    """
    Maps a vector `u` from the unit hypercube [0, 1]^d to the target parameter space.
    (Commonly used as the Prior Transform in nested sampling).
    """
    if hasattr(d, 'icdf') and callable(d.icdf):
        return d.icdf(u)
    elif isinstance(d, dist.Transformed):
        base_x = distribution_hypercube_to_physical(d.distribution, u)
        return d.bijector.forward(base_x) 
    else:
        raise NotImplementedError(
            f"Analytical hypercube mapping is not yet supported for {type(d)}. "
            f"Ensure the distribution has a .icdf() method or is wrapped in a Bijector."
        )

def distribution_physical_to_hypercube(d: dist.AbstractDistribution, x: jnp.ndarray) -> jnp.ndarray:
    """
    Maps a vector `x` from the physical parameter space back to the unit hypercube [0, 1]^d.
    (The mathematical inverse of the hypercube_to_physical mapping).
    """
    if hasattr(d, 'cdf') and callable(d.cdf):
        return d.cdf(x)
    elif isinstance(d, dist.Transformed):
        base_x = d.bijector.inverse(x)
        return distribution_physical_to_hypercube(d.distribution, base_x)
    else:
        raise NotImplementedError(
            f"Analytical physical-to-hypercube mapping is not supported for {type(d)}. "
            f"Ensure the distribution has a .cdf() method or is wrapped in a Bijector."
        )

def unwrap_distribution(distribution: dist.AbstractDistribution) -> tuple[dist.AbstractDistribution, bij.AbstractBijector]:
    """
    Recursively unwraps a potentially Transformed distribution.
    Returns the core base distribution and a bijector
    that transforms the base distribution to the supplied distribution.
    """
    if not isinstance(distribution, dist.Transformed):
        return distribution, bij.Identity()
    elif isinstance(distribution, dist.Transformed) and not isinstance(distribution.distribution, dist.Transformed):
        return distribution.distribution, distribution.bijector
    
    base_dist = distribution
    bijectors = []
    while isinstance(base_dist, dist.Transformed):
        bijectors.append(base_dist.bijector)
        base_dist = base_dist.distribution

    chain = bij.Chain(bijectors)
    return base_dist, chain

def build_batched_mvn(loc, cov):
    batch_ndims = loc.ndim - 1
    builder = dist.MultivariateNormalFullCovariance
    for _ in range(batch_ndims):
        builder = eqx.filter_vmap(builder)
    return builder(loc=loc, covariance_matrix=cov)

def build_batched_normal(loc, scale):
    batch_ndims = loc.ndim
    builder = dist.Normal
    for _ in range(batch_ndims):
        builder = eqx.filter_vmap(builder)
    return builder(loc=loc, scale=scale)


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