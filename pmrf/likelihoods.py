from typing import Callable
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

class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood with variance `noise`.
    
    `noise` must be broadcastable to `y_pred`. The shapes can be arbitrary 
    combinations (e.g., scalar, (nfreq,), (nfreq, nports, nports)), as long as 
    they follow standard NumPy broadcasting rules.
    
    For complex gaussian likelihoods, see :class:`pmrf.likelihoods.ComplexGaussianLikelihood`.
    """
    noise: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray]

    def __call__(self, y_pred: jnp.ndarray | dist.AbstractDistribution, **kwargs):
        if jnp.iscomplexobj(y_pred):
            raise TypeError("`GaussianLikelihood` does not support complex model features. Use `ComplexGaussianLikelihood` instead.")
        
        noise = self.noise(y_pred) if callable(self.noise) else self.noise
        
        if isinstance(y_pred, dist.AbstractDistribution):
            loc = y_pred.mean()
            cov = y_pred.covariance()
            
            # 1. Broadcast noise to match the exact shape of loc (..., k)
            noise_b = jnp.broadcast_to(noise, loc.shape)
            
            # 2. Convert the broadcasted variance into diagonal covariance blocks (..., k, k)
            noise_diag = jnp.vectorize(jnp.diag, signature='(n)->(n,n)')(noise_b)
            
            # 3. Marginalize
            marginal_cov = cov + noise_diag
            
            return dist.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=marginal_cov)
            
        elif isinstance(y_pred, jnp.ndarray):
            # dist.Normal handles broadcasting of `loc` and `scale` natively
            scale = jnp.sqrt(noise)
            return dist.Normal(loc=y_pred, scale=scale)
            
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
       
    `noise` components must be broadcastable to `y_pred`. Supports arbitrary shapes 
    like (nfreq, nports, nports) as long as broadcasting aligns.
    """
    noise: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]]

    def __call__(self, y_pred: jnp.ndarray | dist.AbstractDistribution, **kwargs):
        if isinstance(y_pred, jnp.ndarray) and not jnp.iscomplexobj(y_pred):
            raise TypeError("`y_pred` must be a complex array for `ComplexGaussianLikelihood`.")
            
        noise_val = self.noise(y_pred) if callable(self.noise) else self.noise
        
        # Unpack based on whether it's circularly symmetric (single array) or general (tuple)
        if isinstance(noise_val, tuple):
            if len(noise_val) != 2:
                raise ValueError("If `noise` returns a tuple, it must contain exactly two elements: (covariance, pseudo_covariance).")
            covariance, pseudo_covariance = noise_val
        else:
            covariance = noise_val
            pseudo_covariance = None

        if isinstance(y_pred, jnp.ndarray):
            # `make_complex_normal` uses standard JAX operators which natively support broadcastable shapes
            return make_complex_normal(loc=y_pred, covariance=covariance, pseudo_covariance=pseudo_covariance)
            
        elif isinstance(y_pred, dist.AbstractDistribution):
            if not is_complex_distribution(y_pred):
                raise TypeError("`y_pred` must be a complex distribution transformed by `R2ToComplex`.")
            
            # The complex mean tells us the underlying shape we need to broadcast to
            loc_complex = y_pred.mean() 
            base_dist = y_pred.distribution
            loc_r2 = base_dist.mean() # Shape: (*loc_complex.shape, 2)
            
            # 1. Extract the R2 covariance of the prediction
            if hasattr(base_dist, "covariance") and callable(base_dist.covariance):
                try:
                    cov_r2 = base_dist.covariance() # Shape: (*loc_complex.shape, 2, 2)
                except NotImplementedError:
                    var_r2 = base_dist.variance()
                    # Vectorize diag safely converts shape (..., 2) to (..., 2, 2)
                    cov_r2 = jnp.vectorize(jnp.diag, signature='(n)->(n,n)')(var_r2)
            else:
                var_r2 = base_dist.variance()
                cov_r2 = jnp.vectorize(jnp.diag, signature='(n)->(n,n)')(var_r2)

            # 2. Broadcast the noise terms to the complex shape before mapping to R2
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
            noise_cov_r2 = jnp.stack([row1, row2], axis=-2) # Shape: (*loc_complex.shape, 2, 2)
            
            # 4. Marginalize in R2 space
            marginal_cov_r2 = cov_r2 + noise_cov_r2
            
            marginal_base_dist = dist.MultivariateNormalFullCovariance(
                loc=loc_r2, 
                covariance_matrix=marginal_cov_r2
            )
            
            from distreqx.bijectors import R2ToComplex
            return dist.Transformed(distribution=marginal_base_dist, bijector=R2ToComplex())
            
        else:
            raise TypeError(f"Unsupported distribution type for ComplexGaussianLikelihood: {type(y_pred)}")