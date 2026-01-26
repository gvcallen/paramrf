from pathlib import Path
import dill
from typing import BinaryIO, TypeVar

import zipfile
import cloudpickle
import jsonpickle
import jax.numpy as jnp
import jax.random as jr
import jax
import equinox as eqx

from pmrf.distributions.trainable import TrainableDistribution
from pmrf.distributions.serializable import SerializableDistribution

def _make_flow(samples: jnp.ndarray, key=None, kind='coupling', **init_kwargs):
    from flowjax.flows import coupling_flow, masked_autoregressive_flow
    from flowjax.distributions import Transformed, Normal
    from flowjax.bijections import RationalQuadraticSpline, Affine, Chain        
    import paramax
    
    # Choose the flow kind
    if kind == 'coupling':
        init_fn = coupling_flow
    elif kind == 'masked_autoregressive':
        init_fn = masked_autoregressive_flow
    else:
        raise Exception(f"Unsupported kind {kind} passed to initialize the FlowJAX flow")
    
    # Setup the flow hyperparameters
    init_kwargs.setdefault('base_dist', Normal(jnp.zeros(samples.shape[1])))
    init_kwargs.setdefault('transformer', RationalQuadraticSpline(knots=10, interval=10))
    init_kwargs.setdefault('nn_width', 64)
    init_kwargs.setdefault('nn_depth', 4)
    init_kwargs.setdefault('invert', False)
    if init_kwargs['base_dist'].shape[0] != samples.shape[1]:
        raise Exception("Base distribution must have shape equal to the number of parameters")        
    
    # Initialize the flow        
    flow = init_fn(
        key,
        **init_kwargs,
    )                
    
    # Augment the flow to also normalize the input data
    min, max = samples.min(axis=0), samples.max(axis=0)
    loc, scale = (min + max) / 2, (max - min) / 2
    norm_layer = paramax.NonTrainable(Affine(loc=loc, scale=scale))
    augmented_bijection = Chain([flow.bijection, norm_layer])
    flow = Transformed(flow.base_dist, augmented_bijection)
    
    init_kwargs['kind'] = kind
    return flow, init_kwargs
    
def _train_flow(flow, samples: jnp.ndarray, key=None, **train_kwargs):
    from flowjax.train import fit_to_data
    from flowjax.distributions import Transformed, Normal
    from flowjax.bijections import RationalQuadraticSpline, Affine, Chain

    train_kwargs.setdefault('learning_rate', 1e-3)
    train_kwargs.setdefault('batch_size', 64)
    train_kwargs.setdefault('max_patience', 10)
    flow, _losses = fit_to_data(key, flow, samples, **train_kwargs)
    
    return flow

class FlowJAXDistribution(TrainableDistribution, SerializableDistribution):
    """
    JAX-native wrapper for FlowJax's `Transformed` class.

    It is assumed that the flow performs an N:N mapping from the base distribution
    to the output distribution.
    """
    def __init__(self, flow, init_kwargs=None):
        from flowjax.distributions import Transformed
        
        self.flow: Transformed = flow        
        self.init_kwargs = init_kwargs
        event_shape = (self.flow.base_dist.shape[0],)
        super().__init__(batch_shape=(), event_shape=event_shape)
        
    def sample(self, key, sample_shape):
        return self.flow.sample(key, sample_shape)

    def log_prob(self, value):
        return self.flow.log_prob(value)

    def icdf(self, u):
        z = jax.scipy.special.ndtri(u)
        return self.flow.bijection.transform(z)
    
    def save(self, target: str | Path) -> None:
        if self.init_kwargs is None:
            raise Exception('Cannot save flow without init_kwargs')
        
        

    @classmethod
    def load(cls, path: str | Path) -> 'FlowJAXDistribution':
        raise NotImplementedError
        
    @classmethod
    def from_samples(cls, samples: jnp.ndarray, key=None, kind='coupling', init_kwargs: dict | None = None, **train_kwargs):        
        init_key, train_key = jr.split(key)
        flow, init_kwargs = _make_flow(init_key, kind=kind, init_kwargs=init_kwargs)
        flow = _train_flow(flow, samples, train_key, **train_kwargs)
        
        return FlowJAXDistribution(flow, init_kwargs=init_kwargs)