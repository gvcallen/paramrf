import jax
import jax.numpy as jnp
import distreqx.distributions as dist
import distreqx.bijectors as bij

from pmrf.core import DiscrepancyModel
import parax as prx
from abc import abstractmethod


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
    def __add__(self, other: Kernel) -> SumKernel:
        return SumKernel(self, other)

    def __mul__(self, other: Kernel) -> ProductKernel:
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
    jitter: float = 1e-8

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
        y_ri = jnp.stack([jnp.real(y_pred), jnp.imag(y_pred)], axis=-1)
        loc = jnp.moveaxis(y_ri, 0, -1)
        
        base_dist = dist.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=K)
        combined_dist = dist.Independent(base_dist, reinterpreted_batch_ndims=loc.ndim - 1)
        
        # Corrected to dist.TransformedDistribution
        return dist.Transformed(combined_dist, bij.R2ToComplex())
    
    
__all__ = [
    "Kernel",
    "SumKernel",
    "ProductKernel",
    "ConstantKernel",
    "RBFKernel",
    "WhiteNoiseKernel",
]