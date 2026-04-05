"""
Statistical likelihood models.
"""

from typing import Callable
import jax
import jax.numpy as jnp
import distreqx.distributions as dist
import parax as prx

from pmrf.core import Likelihood
from pmrf.utils import make_complex_normal

def is_complex_array(x):
    return isinstance(x, jnp.ndarray) and jnp.iscomplexobj(x)

def is_complex_distribution(x):
    from distreqx.bijectors import R2ToComplex
    return isinstance(x, dist.Transformed) and isinstance(x.bijector, R2ToComplex)

def _batched_mvn(loc: jnp.ndarray, cov: jnp.ndarray) -> dist.MultivariateNormalFullCovariance:
    """
    Safely initializes a distreqx MVN across arbitrary batch dimensions.
    Vmaps the constructor to cleanly bypass strict `ndim == 1` checks.
    """
    batch_ndims = loc.ndim - 1 # The last dimension is the event
    init_fn = dist.MultivariateNormalFullCovariance
    for _ in range(batch_ndims):
        init_fn = jax.vmap(init_fn)
    return init_fn(loc, cov)

class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood with variance `noise`.
    
    `noise` must be broadcastable to `y_pred`. The shapes can be arbitrary 
    combinations (e.g., scalar, (nfreq,), (nfreq, nports, nports)), as long as 
    they follow standard NumPy broadcasting rules.
    """
    noise: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray]

    def __call__(self, y_pred: jnp.ndarray | dist.AbstractDistribution, **kwargs):
        if jnp.iscomplexobj(y_pred):
            raise TypeError("`GaussianLikelihood` does not support complex model features. Use `ComplexGaussianLikelihood` instead.")
        
        noise_val = self.noise(y_pred) if callable(self.noise) else self.noise
        
        if isinstance(y_pred, jnp.ndarray):
            scale = jnp.sqrt(noise_val)
            return dist.Normal(loc=y_pred, scale=scale)
            
        elif isinstance(y_pred, dist.AbstractDistribution):
            loc = y_pred.mean()
            noise_b = jnp.broadcast_to(noise_val, loc.shape)
            
            # Fast-path for independent Gaussian predictions
            if isinstance(y_pred, dist.Normal):
                marginal_scale = jnp.sqrt(y_pred.variance() + noise_b)
                return dist.Normal(loc=loc, scale=marginal_scale)
                
            elif isinstance(y_pred, dist.Independent) and isinstance(y_pred.distribution, dist.Normal):
                marginal_scale = jnp.sqrt(y_pred.distribution.variance() + noise_b)
                new_base = dist.Normal(loc=y_pred.distribution.mean(), scale=marginal_scale)
                return dist.Independent(new_base)
                
            else:
                # Fallback: full covariance tracking
                cov = y_pred.covariance()
                # Vectorize diag safely converts shape (..., k) to (..., k, k)
                noise_diag = jnp.vectorize(jnp.diag, signature='(n)->(n,n)')(noise_b)
                marginal_cov = cov + noise_diag
                return _batched_mvn(loc, marginal_cov)
                
        else:
            raise TypeError(f"Unsupported distribution type for GaussianLikelihood: {type(y_pred)}")


class ComplexGaussianLikelihood(Likelihood):
    """
    Complex Gaussian likelihood with variance/covariance defined by `noise`.
    
    The noise model can return either:
    1. A single JAX array: Represents the real-valued variance (Hermitian covariance) 
       for a circularly-symmetric complex Gaussian.
    2. A tuple of two JAX arrays: Represents (covariance, pseudo_covariance) for a 
       general complex Gaussian.
    """
    noise: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]]

    def __call__(self, y_pred: jnp.ndarray | dist.AbstractDistribution, **kwargs):
        if isinstance(y_pred, jnp.ndarray) and not jnp.iscomplexobj(y_pred):
            raise TypeError("`y_pred` must be a complex array for `ComplexGaussianLikelihood`.")
            
        noise_val = self.noise(y_pred) if callable(self.noise) else self.noise
        
        if isinstance(noise_val, tuple):
            if len(noise_val) != 2:
                raise ValueError("If `noise` returns a tuple, it must contain exactly two elements.")
            covariance, pseudo_covariance = noise_val
        else:
            covariance = noise_val
            pseudo_covariance = None

        if isinstance(y_pred, jnp.ndarray):
            return make_complex_normal(loc=y_pred, covariance=covariance, pseudo_covariance=pseudo_covariance)
            
        elif isinstance(y_pred, dist.AbstractDistribution):
            if not is_complex_distribution(y_pred):
                raise TypeError("`y_pred` must be a complex distribution transformed by `R2ToComplex`.")
            
            loc_complex = y_pred.mean() 
            base_dist = y_pred.distribution
            loc_r2 = base_dist.mean() # Shape: (*loc_complex.shape, 2)
            
            # 1. Extract the R2 covariance of the prediction
            if hasattr(base_dist, "covariance") and callable(base_dist.covariance):
                try:
                    cov_r2 = base_dist.covariance()
                except NotImplementedError:
                    var_r2 = base_dist.variance()
                    cov_r2 = jnp.vectorize(jnp.diag, signature='(n)->(n,n)')(var_r2)
            else:
                var_r2 = base_dist.variance()
                cov_r2 = jnp.vectorize(jnp.diag, signature='(n)->(n,n)')(var_r2)

            # 2. Broadcast the noise terms
            cov_b = jnp.broadcast_to(covariance, loc_complex.shape)
            gamma = jnp.real(cov_b)
            
            if pseudo_covariance is None:
                c_real = jnp.zeros_like(gamma)
                c_imag = jnp.zeros_like(gamma)
            else:
                pseudo_b = jnp.broadcast_to(pseudo_covariance, loc_complex.shape)
                c_real = jnp.real(pseudo_b)
                c_imag = jnp.imag(pseudo_b)
            
            # 3. Construct the R2 covariance block for the noise
            cov_11 = 0.5 * (gamma + c_real)
            cov_22 = 0.5 * (gamma - c_real)
            cov_12 = 0.5 * c_imag
            
            row1 = jnp.stack([cov_11, cov_12], axis=-1)
            row2 = jnp.stack([cov_12, cov_22], axis=-1)
            noise_cov_r2 = jnp.stack([row1, row2], axis=-2) 
            
            # 4. Marginalize in R2 space
            marginal_cov_r2 = cov_r2 + noise_cov_r2
            
            # Safely batched MVN instantiation
            marginal_base_dist = _batched_mvn(loc_r2, marginal_cov_r2)
            
            from distreqx.bijectors import R2ToComplex
            return dist.Transformed(distribution=marginal_base_dist, bijector=R2ToComplex())
            
        else:
            raise TypeError(f"Unsupported distribution type for ComplexGaussianLikelihood: {type(y_pred)}")