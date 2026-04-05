import distreqx.distributions as dist
import jax.numpy as jnp
import parax as prx

class DiscrepancyModel(prx.Module):
    """
    Base class for discrepancy models.
    
    A discrepancy model maps a model prediction to an updated model prediction.
    This updated prediction can either be deterministic (e.g. a neural network)
    or probabilistic (e.g. a Gaussian process) by either return a JAX array
    or a probability distribution.
    
    These models can then either be used deterministically, or to account for
    the uncertainty in the model itself in conjuction with a likelihood function,
    for example using :class:`pmrf.discrepancy_models.GaussianProcessDiscrepancy
    with :class:`pmrf.likelihoods.ComplexGaussianLikehood`.
    
    See :mod:`pmrf.discrepancy_models` for built-in discrepancy models.
    """
    def __call__(self, y_pred: jnp.ndarray) -> jnp.ndarray | dist.AbstractDistribution:
        raise NotImplementedError