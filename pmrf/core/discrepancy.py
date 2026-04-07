from abc import abstractmethod

import distreqx.distributions as dist
import jax.numpy as jnp
import parax as prx

class DiscrepancyModel(prx.Module):
    """
    Base class for discrepancy models.
    
    A discrepancy model maps a model prediction  to an updated model prediction.
    This updated prediction can either be deterministic (e.g. a neural network)
    or probabilistic (e.g. a Gaussian process) by either returning a JAX array
    or a probability distribution.
    
    These models can then either be used deterministically, or to account for
    the uncertainty in the model itself in conjuction with a likelihood function,
    for example using :class:`pmrf.discrepancy_models.GaussianProcessDiscrepancy
    with :class:`pmrf.likelihoods.ComplexGaussianLikehood`.
    
    NB: Discrepancy models in "event space", where frequency is the generally last axis.
    
    See :mod:`pmrf.discrepancy_models` for built-in discrepancy models.
    """
    @abstractmethod
    def __call__(self, y_event: jnp.ndarray) -> jnp.ndarray | dist.AbstractDistribution:
        raise NotImplementedError
    
    
class Kernel(prx.Module):
    """
    Abstract base class for covariance kernel functions enabling gaussian processes.
    """
    def __add__(self, other: 'Kernel') -> 'Kernel':
        from pmrf.discrepancy_models import SumKernel
        return SumKernel(self, other)

    def __mul__(self, other: 'Kernel') -> 'Kernel':
        from pmrf.discrepancy_models import ProductKernel
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