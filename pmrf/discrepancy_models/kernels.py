"""
Covariance kernels for Gaussian process discrepancy models.
"""
import jax.numpy as jnp

from pmrf.core import CovarianceKernel
import parax as prx


class SumKernel(CovarianceKernel):
    """
    Kernel representing the sum of two kernels.

    Attributes
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    k1: CovarianceKernel
    k2: CovarianceKernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) + self.k2(x1, x2)


class ProductKernel(CovarianceKernel):
    """
    Kernel representing the product of two kernels.

    Attributes
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    k1: CovarianceKernel
    k2: CovarianceKernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) * self.k2(x1, x2)


class ConstantKernel(CovarianceKernel):
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


class RBFKernel(CovarianceKernel):
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
    
    
class PeriodicKernel(CovarianceKernel):
    """
    Periodic (Exp-Sine-Squared) kernel.
    
    Models functions that repeat over a specific period.

    Attributes
    ----------
    period : prx.Parameter
        The period of the kernel, dictating the distance between repetitions (default 1.0).
    length_scale : prx.Parameter
        Characteristic length scale of the correlation (default 1.0).
    """
    period: prx.Parameter = 1.0
    length_scale: prx.Parameter = 1.0

    def __call__(self, x1, x2, key=None):
        # Add a tiny jitter to the squared distance before taking the square root.
        # This prevents NaN gradients during backpropagation when x1 == x2.
        sq_dist = jnp.sum((x1 - x2)**2)
        dist = jnp.sqrt(sq_dist + 1e-12)
        
        # Calculate the periodic component
        arg = jnp.pi * dist / self.period
        sin_term = jnp.sin(arg)
        
        return jnp.exp(-2.0 * (sin_term / self.length_scale)**2)


class WhiteNoiseKernel(CovarianceKernel):
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