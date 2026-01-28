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

class FlowJAXDistribution(TrainableDistribution, SerializableDistribution):
    """
    JAX-native wrapper for FlowJax's `Transformed` class.

    It is assumed that the flow performs an N:N mapping from the base distribution
    to the output distribution.
    """
    def __init__(self, flow, hyper_params=None):
        from flowjax.distributions import Transformed
        
        self.flow: Transformed = flow        
        self.hyper_params = hyper_params
        event_shape = (self.flow.base_dist.shape[0],)
        super().__init__(batch_shape=(), event_shape=event_shape)
        
    def sample(self, key, sample_shape):
        return self.flow.sample(key, sample_shape)

    def log_prob(self, value):
        return self.flow.log_prob(value)

    def icdf(self, u):
        z = jax.scipy.special.ndtri(u)
        return self.flow.bijection.transform(z)
    
    def save(self, target: str | BinaryIO) -> None:
        if not isinstance(target, str):
            return self.write(target)
        
        if self.hyper_params is None:
            raise Exception('Cannot save flow without hyper_params')
        
        target = Path(target)
        
        # We use a ZipFile to combine the json config and binary weights into one file
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            frozen_params = jsonpickle.encode(self.hyper_params)
            zf.writestr('hyper_params.json', frozen_params)
            
            with zf.open('weights.eqx', 'w') as f:
                eqx.tree_serialise_leaves(f, self.flow)

    @classmethod
    def load(cls, source: str | BinaryIO) -> 'FlowJAXDistribution':
        if not isinstance(source, str):
            return cls.read(source)
        
        source = Path(source)
        
        if not source.exists():
            raise FileNotFoundError(f"Could not find flow file: {source}")

        with zipfile.ZipFile(source, 'r') as zf:
            hp_json = zf.read('hyper_params.json').decode('utf-8')
            hyper_params = jsonpickle.decode(hp_json)

            key = jr.key(0)
            skeleton_flow, hyper_params = _make_flow(key, **hyper_params)

            with zf.open('weights.eqx', 'r') as f:
                flow = eqx.tree_deserialise_leaves(f, skeleton_flow)

        return FlowJAXDistribution(flow, hyper_params=hyper_params)
        
    @classmethod
    def from_samples(cls, samples: jnp.ndarray, weights: jnp.ndarray | None = None, key=None, kind='coupling', transformer_cls=None, transformer_kwargs=None, init_kwargs: dict | None = None, **train_kwargs):        
        if key is None:
            key = jr.key(0)
        init_key, train_key = jr.split(key)
        
        theta_min, theta_max, num_params = samples.min(axis=0), samples.max(axis=0), samples.shape[1]
        flow, hyper_params = _make_flow(
            init_key,
            theta_min=theta_min,
            theta_max=theta_max,
            num_params=num_params,
            kind=kind,
            transformer_cls=transformer_cls,
            transformer_kwargs=transformer_kwargs,
            init_kwargs=init_kwargs,
        )
        flow = _train_flow(train_key, flow, samples, weights=weights, **train_kwargs)
        return FlowJAXDistribution(flow, hyper_params=hyper_params)
    
def _make_flow(key, theta_min=None, theta_max=None, num_params=None, kind=None, transformer_cls=None, transformer_args=None, transformer_kwargs=None, init_kwargs: dict | None = None):
    from flowjax.flows import coupling_flow, masked_autoregressive_flow
    from flowjax.distributions import Transformed, Normal
    import flowjax.bijections as fjb
    import paramax

    # We allow a transformer to be passed if it is Affine or RationalQuadaraticSpline
    if init_kwargs is not None and 'transformer' in init_kwargs:
        transformer = init_kwargs.pop('transformer')
        if isinstance(transformer, fjb.RationalQuadraticSpline):
            transformer_cls = 'RationalQuadraticSpline'
            transformer_kwargs = {'knots': transformer.knots, 'interval': transformer.interval, 'softmax_adjust': transformer.softmax_adjust, 'min_derivative': transformer.min_derivative}
        elif isinstance(transformer, fjb.Affine):
            transformer_cls = 'Affine'
            transformer_kwargs = {'loc': transformer.loc, 'scale': transformer.scale}
        else:
            raise Exception("Only RationalQuadraticSpline and Affine supported as transformers for FlowJAXDistribution")


    # Set defaults
    kind = kind if kind is not None else 'coupling'
    init_kwargs = init_kwargs if init_kwargs is not None else {}
    transformer_cls = transformer_cls if transformer_cls is not None else 'RationalQuadraticSpline'
    transformer_args = transformer_args if transformer_args is not None else ()
    transformer_kwargs = transformer_kwargs if transformer_kwargs is not None else {'knots': 10, 'interval': 10}
    
    # Validate input
    if theta_min is None or theta_max is None or num_params is None:
        raise Exception('Number of parameters and bounds must be passed to construct a flow')
    if 'base_dist' in init_kwargs:
        raise Exception("base_dist not supporterd as an init param in FlowJAXDistribution init_kwargs")

    # Choose the flow function
    if kind == 'coupling':
        init_fn = coupling_flow
    elif kind == 'masked_autoregressive':
        init_fn = masked_autoregressive_flow
    else:
        raise Exception(f"Unsupported kind {kind} passed to initialize the FlowJAX flow")
    
    # Setup the flow init parameters
    transformer = getattr(fjb, transformer_cls)(*transformer_args, **transformer_kwargs)
    init_kwargs.setdefault('base_dist', Normal(jnp.zeros(num_params)))
    init_kwargs.setdefault('transformer', transformer)
    init_kwargs.setdefault('nn_width', 64)
    init_kwargs.setdefault('nn_depth', 4)
    init_kwargs.setdefault('invert', False)
    if init_kwargs['base_dist'].shape[0] != num_params:
        raise Exception("Base distribution must have shape equal to the number of parameters")        
    
    # Initialize the flow        
    flow = init_fn(
        key,
        **init_kwargs,
    )                
    
    # Augment the flow to also normalize the input data
    loc, scale = (theta_min + theta_max) / 2, (theta_max - theta_min) / 2
    norm_layer = paramax.NonTrainable(fjb.Affine(loc=loc, scale=scale))
    augmented_bijection = fjb.Chain([flow.bijection, norm_layer])
    flow = Transformed(flow.base_dist, augmented_bijection)

    # Setup and return the hyper-parameters to be saved
    init_kwargs.pop('base_dist')
    init_kwargs.pop('transformer')
    hyper_params = {}
    hyper_params['theta_min'] = theta_min
    hyper_params['theta_max'] = theta_max
    hyper_params['num_params'] = num_params
    hyper_params['kind'] = kind
    hyper_params['transformer_cls'] = transformer_cls
    hyper_params['transformer_args'] = transformer_args
    hyper_params['transformer_kwargs'] = transformer_kwargs
    hyper_params['init_kwargs'] = init_kwargs
    return flow, hyper_params    

def _train_flow(key, flow, samples: jnp.ndarray, weights: jnp.ndarray | None = None, **train_kwargs):
    from flowjax.train import fit_to_data
    
    try:
        from paramax import unwrap 
    except ImportError:
        from flowjax.distributions import unwrap

    train_kwargs.setdefault('learning_rate', 1e-3)
    train_kwargs.setdefault('batch_size', 64)
    train_kwargs.setdefault('max_patience', 10)

    if weights is not None:
        # Scale weights so they sum to N (the number of samples).
        # This prevents gradients from vanishing if weights are normalized to 1.0
        n_samples = weights.shape[0]
        sum_weights = weights.sum()
        
        # Avoid division by zero
        scale_factor = n_samples / (sum_weights + 1e-10)
        weights = weights * scale_factor

        def weighted_loss_fn(params, static, x, w, key=None):
            dist = unwrap(eqx.combine(params, static))
            # We use sum() here because we divide by the batch size 
            # implicit in the optimizer or loop logic, effectively calculating the mean.
            # reducing by mean over a batch of weighted samples preserves the scale.
            return -(w * dist.log_prob(x)).mean()

        data = (samples, weights)
        train_kwargs["loss_fn"] = weighted_loss_fn
    else:
        data = samples

    flow, _losses = fit_to_data(key, flow, data, **train_kwargs)
    
    return flow