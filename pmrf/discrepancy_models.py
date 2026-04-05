"""
Models that cater for the discrepancy between a physical model and data.
"""

import jax
import jax.numpy as jnp
import distreqx.distributions as dist
import distreqx.bijectors as bij

from pmrf.core import DiscrepancyModel
import parax as prx
from abc import abstractmethod

bij.Reshape()

def _fmt(val: jnp.ndarray) -> str:
    """Helper to format JAX arrays as clean strings for printing."""
    if hasattr(val, 'item') and val.ndim == 0:
        return f"{val.item():.3g}"
    if hasattr(val, 'tolist'):
        return str([float(f"{x:.3g}") for x in val.flatten()])
    return str(val)


class Kernel(prx.Module):
    """
    Abstract base class for kernel functions enabling kernel algebra.
    """
    def __add__(self, other: 'Kernel') -> 'SumKernel':
        return SumKernel(self, other)

    def __mul__(self, other: 'Kernel') -> 'ProductKernel':
        return ProductKernel(self, other)

    @abstractmethod
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray, key=None) -> jnp.ndarray:
        """
        Evaluate the kernel between two points.

        Parameters
        ----------
        x1 : jnp.ndarray
            First input point.
        x2 : jnp.ndarray
            Second input point.
        key : jax.random.PRNGKey, optional
            Random key for stochastic kernels.

        Returns
        -------
        jnp.ndarray
            Kernel covariance scalar.
        """
        pass


class SumKernel(Kernel):
    """
    Kernel representing the sum of two kernels.

    Attributes
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    k1: Kernel
    k2: Kernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) + self.k2(x1, x2)


class ProductKernel(Kernel):
    """
    Kernel representing the product of two kernels.

    Attributes
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    k1: Kernel
    k2: Kernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) * self.k2(x1, x2)


class ConstantKernel(Kernel):
    """
    Kernel that returns a constant variance.

    Attributes
    ----------
    variance : prx.Parameter
        Constant variance value (default 1.0).
    """
    variance: prx.Parameter = 1.0

    def __call__(self, x1, x2, key=None):
        return self.variance


class RBFKernel(Kernel):
    """
    Radial Basis Function (Squared Exponential) kernel.

    Attributes
    ----------
    length_scale : prx.Parameter
        Characteristic length scale of the correlation (default 1.0).
    """
    length_scale: prx.Parameter = 1.0

    def __call__(self, x1, x2, key=None):
        scaled_diff = (x1 - x2) / self.length_scale
        sq_dist = jnp.sum(scaled_diff**2)
        return jnp.exp(-0.5 * sq_dist)


class WhiteNoiseKernel(Kernel):
    """
    Kernel representing independent Gaussian noise.

    Attributes
    ----------
    variance : prx.Parameter
        Noise variance level (default 1.0).
    """
    variance: prx.Parameter = 1.0

    def __call__(self, x1, x2, key=None):
        is_equal = jnp.allclose(x1, x2)
        return jnp.where(is_equal, self.variance, 0.0)


class GaussianProcessDiscrepancy(DiscrepancyModel):
    """
    Maps model predictions to a Gaussian Process distribution over frequency.

    Attributes
    ----------
    kernel : Kernel
        The kernel governing frequency-domain correlation.
    jitter : float
        Small value added to the Gram matrix diagonal for numerical stability.
    """
    kernel: Kernel
    jitter: float = 1e-10

    def __call__(self, y_pred: jnp.ndarray, x: jnp.ndarray) -> dist.AbstractDistribution:
        """
        Transforms a deterministic prediction into a GP-based distribution.

        Parameters
        ----------
        y_pred : jnp.ndarray
            Deterministic model predictions of shape (N, ...).
        x : jnp.ndarray
            Frequency features of shape (N,) or (N, F).

        Returns
        -------
        dist.AbstractDistribution
            A distribution (MVN or Transformed) over the model output.
        """
        if x.ndim == 1:
            x_feat = x[:, jnp.newaxis]
        else:
            x_feat = x
            
        v_kern = jax.vmap(jax.vmap(self.kernel, in_axes=(None, 0)), in_axes=(0, None))
        K = v_kern(x_feat, x_feat)
        K = K + jnp.eye(x.shape[0]) * self.jitter

        if jnp.iscomplexobj(y_pred):
            return self._handle_complex(y_pred, K)
        else:
            return self._handle_real(y_pred, K)

    def _handle_real(self, y_pred: jnp.ndarray, K: jnp.ndarray) -> dist.AbstractDistribution:
        """
        Internal handler for real-valued model outputs.

        Parameters
        ----------
        y_pred : jnp.ndarray
            Real predictions.
        K : jnp.ndarray
            Computed Gram matrix.

        Returns
        -------
        dist.Independent
            A batch of independent MVNs over the feature dimensions.
        """
        loc = jnp.moveaxis(y_pred, 0, -1)
        base_dist = dist.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=K)
        return dist.Independent(base_dist, reinterpreted_batch_ndims=loc.ndim - 1)

    def _handle_complex(self, y_pred: jnp.ndarray, K: jnp.ndarray) -> dist.AbstractDistribution:
        """
        Internal handler for complex-valued model outputs using R2ToComplex.

        Parameters
        ----------
        y_pred : jnp.ndarray
            Complex predictions.
        K : jnp.ndarray
            Computed Gram matrix.

        Returns
        -------
        dist.TransformedDistribution
            A complex-valued distribution mapped via bijector.
        """
        N = y_pred.shape[-1]
        
        # 1. Stack Real/Imag to create an array of shape (2, N)
        y_ri = jnp.stack([jnp.real(y_pred), jnp.imag(y_pred)], axis=0)
        
        # 2. Vmap the MVN constructor to create a batch of 2 distributions
        batched_mvn = jax.vmap(dist.MultivariateNormalFullCovariance, in_axes=(0, None))(y_ri, K)
        
        # 3. Fuse the batch into a single event of shape (2, N)
        indep_dist = dist.Independent(batched_mvn)
        
        # 4. Use your existing Reshape to flatten to (2N,)
        import numpy as np
        b_flatten = bij.Reshape(in_shape=(2, N), out_shape=(2 * N,))
        
        # 5. Build a permutation array that interleaves Reals and Imags
        # This maps [R0, R1, I0, I1] -> [R0, I0, R1, I1]
        interleave_perm = np.arange(2 * N).reshape((2, N)).T.flatten()
        b_permute = bij.Permute(permutation=interleave_perm)
        
        # 6. Use your existing Reshape to group into (N, 2)
        b_regroup = bij.Reshape(in_shape=(2 * N,), out_shape=(N, 2))
        
        # 7. Chain them together
        d = dist.Transformed(indep_dist, b_flatten)
        d = dist.Transformed(d, b_permute)
        d = dist.Transformed(d, b_regroup)
        
        return dist.Transformed(d, bij.R2ToComplex())
    
__all__ = [
    "Kernel",
    "SumKernel",
    "ProductKernel",
    "ConstantKernel",
    "RBFKernel",
    "WhiteNoiseKernel",
    "GaussianProcessDiscrepancy",
]