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
    lengthscale : prx.Parameter
        Characteristic length scale of the correlation (default 1.0).
    """
    lengthscale: prx.Parameter = 1.0

    def __call__(self, x1, x2, key=None):
        scaled_diff = (x1 - x2) / self.lengthscale
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
    lengthscale : prx.Parameter
        Characteristic length scale of the correlation (default 1.0).
    """
    period: prx.Parameter = 1.0
    lengthscale: prx.Parameter = 1.0

    def __call__(self, x1, x2, key=None):
        # Add a tiny jitter to the squared distance before taking the square root.
        # This prevents NaN gradients during backpropagation when x1 == x2.
        sq_dist = jnp.sum((x1 - x2)**2)
        dist = jnp.sqrt(sq_dist + 1e-12)
        
        # Calculate the periodic component
        arg = jnp.pi * dist / self.period
        sin_term = jnp.sin(arg)
        
        return jnp.exp(-2.0 * (sin_term / self.lengthscale)**2)


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
    
    
class AutoCrossKernel(CovarianceKernel):
    """
    Kernel that routes between a auto-correlation and cross-correlation kernels.
    
    Constructs a block matrix where diagonal elements evaluate the auto kernel
    and off-diagonal elements evaluate the cross kernel.

    This can be used to model reflection (auto) and transmission (cross) discrepancy separately.
    In this case, set `num_outputs` to the number of ports.
    """
    auto: CovarianceKernel
    cross: CovarianceKernel
    num_outputs: int = prx.field(static=True)

    def __call__(self, x1, x2, key=None):
        # Evaluate both underlying kernels
        val_gamma = self.auto(x1, x2, key=key)
        val_tau = self.cross(x1, x2, key=key)
        
        # Create a 2D boolean mask for the diagonal
        eye = jnp.eye(self.num_outputs, dtype=bool)
        
        # If the base kernels return batched arrays (e.g., shape (2,) for real/imag),
        # we need to append dummy dimensions to the mask so jnp.where broadcasts correctly.
        # This results in a mask shape of (nports, nports, 1, ..., 1)
        eye_shape = [self.num_outputs, self.num_outputs] + [1] * val_gamma.ndim
        eye_broadcastable = eye.reshape(eye_shape)

        # Route the kernel evaluations
        return jnp.where(eye_broadcastable, val_gamma, val_tau)
    

class SharedIndependentKernel(CovarianceKernel):
    """
    Evaluates a base kernel and broadcasts its output to represent 
    multiple independent dimensions (e.g., real and imaginary parts) 
    withed share hyperparameters.

    Attributes
    ----------
    base_kernel : CovarianceKernel
        The underlying kernel whose parameters are shared.
    output_shape : tuple
        The shape of the independent outputs to broadcast to.
    """
    base_kernel: CovarianceKernel = prx.field(transparent=True)
    output_shape: tuple = prx.field(static=True)

    def __call__(self, x1, x2, key=None):
        # Evaluate the underlying shared kernel
        val = self.base_kernel(x1, x2, key=key)
        
        # Broadcast the evaluation to the target shape.
        # By appending the output_shape to the end of val.shape, 
        # this safely handles both scalar evaluations and already-batched evaluations.
        target_shape = val.shape + self.output_shape
        
        return jnp.broadcast_to(val, target_shape)
    

class ZeroKernel(CovarianceKernel):
    """
    Kernel that always evaluates to zero.

    Useful for masking out cross-covariances in multi-output models
    to enforce strict independence between tasks.
    """

    def __call__(self, x1, x2, key=None):
        return jnp.asarray(0.0)