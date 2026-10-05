"""
Covariance kernels for Gaussian processes.

Useful for discrepancy modeling. See :mod:`pmrf.discrepancy_models`
for more details.
"""
from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from pmrf.utils import field, unwrap
from pmrf.types import ArrayLike
from pmrf.modules.base import Module


def gram(
    kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    *,
    jitter: float = 0.0,
) -> jnp.ndarray:
    """
    Build the Gram (covariance) matrix of a kernel evaluated at a set of inputs.

    The kernel is evaluated for every pair of input points using a double
    :func:`jax.vmap`, producing an ``(N, N)`` matrix. Batching is preserved
    exactly: if the kernel returns an array of shape ``batch_shape`` for a
    single pair of points (for example a kernel whose parameters have shape
    ``(D,)``), the result has shape ``(*batch_shape, N, N)``.

    Parameters
    ----------
    kernel : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        The covariance kernel. Accepts two input points of identical shape and
        returns an array broadcastable to the kernel's batch shape.
        Can be a function or a callable PyTree.
        See :mod:`pmrf.covariance_kernels` for built-in covariance kernels.
    x : jnp.ndarray
        The input points. An array of shape ``(N,)`` is treated as ``N``
        one-dimensional features; an array of shape ``(N, d)`` is treated as
        ``N`` ``d``-dimensional features.
    jitter : float, default=0.0
        A small scalar added to the diagonal of the matrix for numerical
        stability. The default of ``0.0`` returns the raw Gram matrix.

    Returns
    -------
    jnp.ndarray
        The Gram matrix, of shape ``(*batch_shape, N, N)``, where
        ``batch_shape`` is the shape returned by the kernel for a single pair.
    """
    # Unwrap the hyperparameters to their values once, behind a barrier. Otherwise
    # XLA can fuse each one's raw-to-value bijector chain into the elementwise loop
    # that builds the matrix and recompute it per entry.
    leaves, structure = eqx.partition(unwrap(kernel), eqx.is_array)
    kernel = eqx.combine(jax.lax.optimization_barrier(leaves), structure)

    x = jnp.asarray(x)
    if x.ndim == 1:
        x_feat = x[:, None]
    else:
        x_feat = x

    inner_vmap = jax.vmap(kernel, in_axes=(None, 0), out_axes=-1)
    outer_vmap = jax.vmap(inner_vmap, in_axes=(0, None), out_axes=-2)

    K = outer_vmap(x_feat, x_feat)
    if jitter:
        K = K + jnp.eye(x_feat.shape[0]) * jitter
    return K


class AbstractCovarianceKernel(Module):
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

    def gram(self, x: jnp.ndarray, *, jitter: float = 0.0) -> jnp.ndarray:
        """
        Build the Gram (covariance) matrix of this kernel evaluated at ``x``.

        Equivalent to ``pmrf.covariance_kernels.gram(self, x, jitter=jitter)``.

        Parameters
        ----------
        x : jnp.ndarray
            The input points, of shape ``(N,)`` or ``(N, d)``.
        jitter : float, default=0.0
            A small scalar added to the diagonal for numerical stability.

        Returns
        -------
        jnp.ndarray
            The Gram matrix, of shape ``(*batch_shape, N, N)``.
        """
        return gram(self, x, jitter=jitter)

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

    Parameters
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    #: First kernel.
    k1: AbstractCovarianceKernel
    
    #: Second kernel.
    k2: AbstractCovarianceKernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) + self.k2(x1, x2)


class ProductKernel(AbstractCovarianceKernel):
    """
    Kernel representing the product of two kernels.

    Parameters
    ----------
    k1 : Kernel
        First kernel operand.
    k2 : Kernel
        Second kernel operand.
    """
    #: First kernel.
    k1: AbstractCovarianceKernel
    
    #: Second kernel.
    k2: AbstractCovarianceKernel

    def __call__(self, x1, x2, key=None):
        return self.k1(x1, x2) * self.k2(x1, x2)

class ConstantKernel(AbstractCovarianceKernel):
    """
    Kernel that returns a constant variance.

    Parameters
    ----------
    variance : ArrayLike
        Constant variance value.
    """
    #: The variance.
    variance: ArrayLike

    def __call__(self, x1, x2, key=None):
        return self.variance


class RBFKernel(AbstractCovarianceKernel):
    """
    Radial Basis Function (Squared Exponential) kernel.

    Parameters
    ----------
    lengthscale : ArrayLike
        Characteristic length scale of the correlation.
    """
    lengthscale: ArrayLike

    def __call__(self, x1, x2, key=None):
        scaled_diff = (x1 - x2) / self.lengthscale
        sq_dist = jnp.sum(scaled_diff**2)
        return jnp.exp(-0.5 * sq_dist)
    
    
class PeriodicKernel(AbstractCovarianceKernel):
    """
    Periodic (Exp-Sine-Squared) kernel.
    
    Models functions that repeat over a specific period.

    Parameters
    ----------
    period : ArrayLike
        The period of the kernel, dictating the distance between repetitions.
    lengthscale : ArrayLike
        Characteristic length scale of the correlation.
    """
    #: The period.
    period: ArrayLike

    #: The lengthscale,
    lengthscale: ArrayLike

    def __call__(self, x1, x2, key=None):
        # Add a tiny jitter to the squared distance before taking the square root.
        # This prevents NaN gradients during backpropagation when x1 == x2.
        sq_dist = jnp.sum((x1 - x2)**2)
        dist = jnp.sqrt(sq_dist + 1e-12)
        
        # Calculate the periodic component
        arg = jnp.pi * dist / self.period
        sin_term = jnp.sin(arg)
        
        return jnp.exp(-2.0 * (sin_term / self.lengthscale)**2)


class CosineKernel(AbstractCovarianceKernel):
    r"""
    Cosine kernel.

    Models zero-mean oscillatory correlation with a single period, which can be
    negative. Unlike :class:`PeriodicKernel`, the correlation goes to $-1$ at
    half a period.

    $$k(x_1, x_2) = \cos\left(\frac{2\pi\, \mathbf{1}^\top (x_1 - x_2)}{p}\right)$$

    where $p$ is the ``period``.

    For $d$-dimensional inputs the oscillation runs along the all-ones direction:
    the differences are summed, not combined into a Euclidean distance, since
    $\cos(2\pi \lVert x_1 - x_2 \rVert / p)$ is not positive semi-definite
    for $d > 1$.

    Its Gram matrix has rank two, so it is singular on its own. Multiply by a
    decaying kernel (e.g. :class:`Matern52Kernel`) for a damped oscillation,
    and by a float for its variance.

    Parameters
    ----------
    period : ArrayLike
        The period of the oscillation. A period of shape ``(D,)`` gives ``D``
        batched kernels, as for :class:`PeriodicKernel`.
    """
    #: The period.
    period: ArrayLike

    def __call__(self, x1, x2, key=None):
        return jnp.cos(2 * jnp.pi * jnp.sum(x1 - x2) / self.period)


class WhiteNoiseKernel(AbstractCovarianceKernel):
    """
    Kernel representing independent Gaussian noise.

    Parameters
    ----------
    variance : ArrayLike
        Noise variance level.
    """
    #: The variance.
    variance: ArrayLike

    def __call__(self, x1, x2, key=None):
        is_equal = jnp.allclose(x1, x2)
        return jnp.where(is_equal, self.variance, 0.0)
    
    
class Matern32Kernel(AbstractCovarianceKernel):
    """
    Matérn kernel with nu=3/2.
    
    This kernel represents functions that are once differentiable, 
    making it less smooth than the RBF kernel and better suited 
    for realistic physical signals.

    Parameters
    ----------
    lengthscale : ArrayLike
        Characteristic length scale of the correlation.
    """
    # The lengthscale
    lengthscale: ArrayLike

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

    Parameters
    ----------
    lengthscale : ArrayLike
        Characteristic length scale of the correlation.
    """
    # The lengthscale.
    lengthscale: ArrayLike

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

    Parameters
    ----------
    auto : AbstractCovarianceKernel
        The covariance kernel describing the auto terms (e.g. S11, S22).
    cross : AbstractCovarianceKernel
        The covariance kernel describing the cross terms (e.g. S21, S43).
    """
    # The auto terms.
    auto: AbstractCovarianceKernel
    
    # The cross terms.
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
    Shares a base kernel across trailing independent batch axes (e.g. the real
    and imaginary parts), so that every element on those axes uses the same
    hyperparameters.

    The shared axes are appended to the base kernel's output as size-1 axes
    rather than materialized. A base output of shape ``batch_shape`` becomes
    ``(*batch_shape, 1, ..., 1)``, which broadcasts to any size along the
    shared axes, so a batched base kernel (e.g. :class:`AutoCrossKernel`)
    keeps routing along its own leading axes.

    Parameters
    ----------
    base_kernel : AbstractCovarianceKernel
        The underlying kernel whose parameters are shared.
    num_shared_axes : int, default=1
        The number of trailing batch axes the base kernel is shared across.
    """
    #: The base kernel.
    base_kernel: AbstractCovarianceKernel

    #: The number of trailing shared axes.
    num_shared_axes: int = field(default=1, static=True)

    def __call__(self, x1, x2, key=None):
        val = jnp.asarray(self.base_kernel(x1, x2, key=key))
        return val.reshape(val.shape + (1,) * self.num_shared_axes)
    

class ZeroKernel(AbstractCovarianceKernel):
    """
    Kernel that always evaluates to zero.

    Useful for masking out cross-covariances in multi-output models
    to enforce strict independence between tasks.
    """
    def __call__(self, x1, x2, key=None):
        return jnp.asarray(0.0)
