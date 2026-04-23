import jax.numpy as jnp
import distreqx.distributions as dist
import distreqx.bijectors as bij

import jax.numpy as jnp

def hypercube_to_physical(d: dist.AbstractDistribution, u: jnp.ndarray) -> jnp.ndarray:
    """
    Maps a vector `u` from the unit hypercube [0, 1]^d to the target parameter space.
    (Commonly used as the Prior Transform in nested sampling).
    """
    if hasattr(d, 'icdf') and callable(d.icdf):
        return d.icdf(u)
    elif isinstance(d, dist.Transformed):
        base_x = hypercube_to_physical(d.distribution, u)
        return d.bijector.forward(base_x) 
    else:
        raise NotImplementedError(
            f"Analytical hypercube mapping is not yet supported for {type(d)}. "
            f"Ensure the distribution has a .icdf() method or is wrapped in a Bijector."
        )

def physical_to_hypercube(d: dist.AbstractDistribution, x: jnp.ndarray) -> jnp.ndarray:
    """
    Maps a vector `x` from the physical parameter space back to the unit hypercube [0, 1]^d.
    (The mathematical inverse of the hypercube_to_physical mapping).
    """
    if hasattr(d, 'cdf') and callable(d.cdf):
        return d.cdf(x)
    elif isinstance(d, dist.Transformed):
        base_x = d.bijector.inverse(x)
        return physical_to_hypercube(d.distribution, base_x)
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