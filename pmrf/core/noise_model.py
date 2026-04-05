import jax.numpy as jnp
import parax as prx

class NoiseModel(prx.Module, prx.Operator):
    """
    Base class for noise models.
    
    A noise model maps a model prediction to a noise parameter, such as variance.
    
    For example, for a real-valued Gaussian likelihood (:class:`pmrf.likelihoods.GaussianLikehood`),
    the noise model can be used with real-valued inputs to model the Gaussian's scale squared.
    
    For complex-valued likelihoods (:class:`pmrf.likelihoods.ComplexGaussianLikehood`), a tuple can
    be returned, representing the Hermitian and pseudo-noise separately.
    """
    def __call__(self, y_pred: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray | jnp.ndarray]:
        raise NotImplementedError