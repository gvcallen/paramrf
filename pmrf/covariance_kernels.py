"""
Covariance kernels for Gaussian process discrepancy models.
"""
from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx

from pmrf.jax_utils import field
from pmrf.parameters import Param, param


class AbstractCovarianceKernel(eqx.Module):
    """
    Abstract base class for covariance kernel functions.
    
    These kernels are used in a Gaussian Process for discrepancy modeling.
    """
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
        raise NotImplementedError

    def __add__(self, other: 'AbstractCovarianceKernel') -> 'AbstractCovarianceKernel':
        from pmrf.covariance_kernels import SumKernel
        return SumKernel(self, other)

    def __mul__(self, other: 'AbstractCovarianceKernel | Param | float') -> 'AbstractCovarianceKernel':
        from pmrf.covariance_kernels import ProductKernel, ConstantKernel
        
        if isinstance(other, AbstractCovarianceKernel):
            return ProductKernel(self, other)    
        else:
            return ProductKernel(self, ConstantKernel(other))

class SumKernel(AbstractCovarianceKernel):
    """
    Kernel representing the sum of two kernels.

    Attributes
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    k1: AbstractCovarianceKernel
    k2: AbstractCovarianceKernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) + self.k2(x1, x2)


class ProductKernel(AbstractCovarianceKernel):
    """
    Kernel representing the product of two kernels.

    Attributes
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    k1: AbstractCovarianceKernel
    k2: AbstractCovarianceKernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) * self.k2(x1, x2)


class ConstantKernel(AbstractCovarianceKernel):
    """
    Kernel that returns a constant variance.

    Attributes
    ----------
    variance : prx.Parameter
        Constant variance value (default 1.0).
    """
    variance: Param = param(1.0)

    def __call__(self, x1, x2, key=None):
        return self.variance


class RBFKernel(AbstractCovarianceKernel):
    """
    Radial Basis Function (Squared Exponential) kernel.

    Attributes
    ----------
    lengthscale : prx.Parameter
        Characteristic length scale of the correlation (default 1.0).
    """
    lengthscale: Param = param(1.0)

    def __call__(self, x1, x2, key=None):
        scaled_diff = (x1 - x2) / self.lengthscale
        sq_dist = jnp.sum(scaled_diff**2)
        return jnp.exp(-0.5 * sq_dist)
    
    
class PeriodicKernel(AbstractCovarianceKernel):
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
    period: Param = param(1.0)
    lengthscale: Param = param(1.0)

    def __call__(self, x1, x2, key=None):
        # Add a tiny jitter to the squared distance before taking the square root.
        # This prevents NaN gradients during backpropagation when x1 == x2.
        sq_dist = jnp.sum((x1 - x2)**2)
        dist = jnp.sqrt(sq_dist + 1e-12)
        
        # Calculate the periodic component
        arg = jnp.pi * dist / self.period
        sin_term = jnp.sin(arg)
        
        return jnp.exp(-2.0 * (sin_term / self.lengthscale)**2)


class WhiteNoiseKernel(AbstractCovarianceKernel):
    """
    Kernel representing independent Gaussian noise.

    Attributes
    ----------
    variance : prx.Parameter
        Noise variance level (default 1.0).
    """
    variance: Param = param(1.0)

    def __call__(self, x1, x2, key=None):
        is_equal = jnp.allclose(x1, x2)
        return jnp.where(is_equal, self.variance, 0.0)
    
    
class Matern32Kernel(AbstractCovarianceKernel):
    """
    Matérn kernel with nu=3/2.
    
    This kernel represents functions that are once differentiable, 
    making it less smooth than the RBF kernel and better suited 
    for realistic physical signals.

    Attributes
    ----------
    lengthscale : prx.Parameter
        Characteristic length scale of the correlation (default 1.0).
    """
    lengthscale: Param = param(1.0)

    def __call__(self, x1, x2, key=None):
        scaled_diff = (x1 - x2) / self.lengthscale
        sq_dist = jnp.sum(scaled_diff**2)
        
        # Add a tiny jitter to the squared distance before taking the square root.
        # This prevents NaN gradients during backpropagation when x1 == x2.
        dist = jnp.sqrt(sq_dist + 1e-12)
        
        sqrt3_dist = jnp.sqrt(3.0) * dist
        return (1.0 + sqrt3_dist) * jnp.exp(-sqrt3_dist)


class Matern52Kernel(AbstractCovarianceKernel):
    """
    Matérn kernel with nu=5/2.
    
    This kernel represents functions that are twice differentiable. 
    It strikes a balance between the rougher Matérn 3/2 and the 
    infinitely smooth RBF kernel.

    Attributes
    ----------
    lengthscale : prx.Parameter
        Characteristic length scale of the correlation (default 1.0).
    """
    lengthscale: Param = param(1.0)

    def __call__(self, x1, x2, key=None):
        scaled_diff = (x1 - x2) / self.lengthscale
        sq_dist = jnp.sum(scaled_diff**2)
        
        # Add a tiny jitter to the squared distance before taking the square root.
        # This prevents NaN gradients during backpropagation when x1 == x2.
        dist = jnp.sqrt(sq_dist + 1e-12)
        
        sqrt5_dist = jnp.sqrt(5.0) * dist
        # Note: We use the sq_dist directly for the squared term to avoid 
        # compounding numerical inaccuracies from the jittered square root
        sq_term = (5.0 / 3.0) * sq_dist 
        
        return (1.0 + sqrt5_dist + sq_term) * jnp.exp(-sqrt5_dist)
    
    
class AutoCrossKernel(AbstractCovarianceKernel):
    """
    Kernel that routes between a auto-correlation and cross-correlation kernels.
    
    Constructs a block matrix where diagonal elements evaluate the auto kernel
    and off-diagonal elements evaluate the cross kernel.

    This can be used to model reflection (auto) and transmission (cross) discrepancy separately.
    In this case, set `num_outputs` to the number of ports.
    """
    auto: AbstractCovarianceKernel
    cross: AbstractCovarianceKernel
    num_outputs: int = field(static=True)

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
    

class SharedIndependentKernel(AbstractCovarianceKernel):
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
    base_kernel: AbstractCovarianceKernel
    output_shape: tuple = field(static=True)

    def __call__(self, x1, x2, key=None):
        # Evaluate the underlying shared kernel
        val = self.base_kernel(x1, x2, key=key)
        
        # Broadcast the evaluation to the target shape.
        # By appending the output_shape to the end of val.shape, 
        # this safely handles both scalar evaluations and already-batched evaluations.
        target_shape = val.shape + self.output_shape
        
        return jnp.broadcast_to(val, target_shape)
    

class ZeroKernel(AbstractCovarianceKernel):
    """
    Kernel that always evaluates to zero.

    Useful for masking out cross-covariances in multi-output models
    to enforce strict independence between tasks.
    """
    def __call__(self, x1, x2, key=None):
        return jnp.asarray(0.0)