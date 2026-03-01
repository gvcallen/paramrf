import io
import base64
from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from numpyro.distributions import Distribution

# =============================================================================
# 1. THE WRAPPER CLASS
# =============================================================================

class FlowJAXDistribution(Distribution):
    """
    JAX-native wrapper for FlowJax's `Transformed` class.
    
    This class wraps a trained FlowJax normalizing flow so that it conforms to 
    the standard NumPyro Distribution interface. It natively supports serialization 
    via `jsonpickle` by delegating PyTree array serialization to Equinox.
    """
    def __init__(self, flow: Any, skeleton_fn: Callable[[], Any]):
        """
        Parameters
        ----------
        flow : Any
            The trained FlowJax `Transformed` distribution (an Equinox module).
        skeleton_fn : Callable[[], Any]
            A zero-argument closure that, when called, returns an uninitialized 
            replica of the flow's architecture. Used during deserialization.
        """
        self.flow = flow
        self.skeleton_fn = skeleton_fn
        
        event_shape = (self.flow.base_dist.shape[0],)
        super().__init__(batch_shape=(), event_shape=event_shape)
        
    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        return self.flow.sample(key, sample_shape)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self.flow.log_prob(value)

    def icdf(self, u: jnp.ndarray) -> jnp.ndarray:
        z = jax.scipy.special.ndtri(u)
        return self.flow.bijection.transform(z)

    def __getstate__(self) -> dict:
        """Package the state natively for jsonpickle."""
        state = self.__dict__.copy()
        
        # Serialize the Equinox arrays to an in-memory byte buffer
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self.flow)
        
        # Convert the binary buffer to a base64 string for JSON compatibility
        state['_flow_bytes'] = base64.b64encode(buffer.getvalue()).decode('ascii')
        
        # Remove the un-picklable Equinox module from the state dictionary
        state.pop('flow', None)
        
        return state

    def __setstate__(self, state: dict) -> None:
        """Unpack the state from jsonpickle."""
        flow_bytes = state.pop('_flow_bytes')
        
        # Restore standard attributes (Crucially, this restores self.skeleton_fn)
        self.__dict__.update(state)
        
        # Decode the base64 string back into a binary buffer
        raw_bytes = base64.b64decode(flow_bytes)
        buffer = io.BytesIO(raw_bytes)
        
        # Reconstruct the Equinox model using the restored closure
        empty_skeleton = self.skeleton_fn()
        self.flow = eqx.tree_deserialise_leaves(buffer, empty_skeleton)

def train_flowjax(
    samples: jnp.ndarray, 
    weights: jnp.ndarray | None = None, 
    key: jnp.ndarray | None = None, 
    kind: str = 'coupling', 
    transformer_cls: str | None = None, 
    transformer_kwargs: dict | None = None, 
    init_kwargs: dict | None = None, 
    **train_kwargs
) -> FlowJAXDistribution:
    """
    Build, train, and wrap a standard FlowJax normalizing flow from empirical samples.

    Parameters
    ----------
    samples : jnp.ndarray
        The training data samples of shape (N, num_params).
    weights : jnp.ndarray | None, optional
        Optional sample weights of shape (N,).
    key : jnp.ndarray | None, optional
        The random key for initialization and training.
    kind : str, optional
        The type of flow ('coupling' or 'masked_autoregressive').
    transformer_cls : str | None, optional
        The type of bijection transformer to use.
    transformer_kwargs : dict | None, optional
        Arguments passed to the transformer.
    init_kwargs : dict | None, optional
        Arguments passed to the flow initialization function.
    **train_kwargs : dict
        Arguments passed to the underlying FlowJax training loop.

    Returns
    -------
    FlowJAXDistribution
        The trained and wrapped FlowJax distribution.
    """
    if key is None:
        key = jr.key(0)
    init_key, train_key = jr.split(key)
    
    theta_min = samples.min(axis=0)
    theta_max = samples.max(axis=0)
    num_params = samples.shape[1]
    
    # 1. Define the picklable architecture closure. 
    # jsonpickle natively saves this function AND the specific variables it references.
    def skeleton_fn():
        return _make_flow(
            jr.key(0), 
            theta_min=theta_min, 
            theta_max=theta_max, 
            num_params=num_params, 
            kind=kind,
            transformer_cls=transformer_cls,
            transformer_kwargs=transformer_kwargs,
            init_kwargs=init_kwargs
        )

    # 2. Build the uninitialized flow architecture
    flow = skeleton_fn()
    
    # 3. Train the flow on the provided data
    trained_flow = _train_flow(train_key, flow, samples, weights=weights, **train_kwargs)
    
    # 4. Wrap and return
    return FlowJAXDistribution(trained_flow, skeleton_fn=skeleton_fn)


# =============================================================================
# 3. INTERNAL UTILITIES
# =============================================================================

def _make_flow(
    key: jnp.ndarray, 
    theta_min: jnp.ndarray, 
    theta_max: jnp.ndarray, 
    num_params: int, 
    kind: str = 'coupling', 
    transformer_cls: str | None = None, 
    transformer_kwargs: dict | None = None, 
    init_kwargs: dict | None = None
) -> Any:
    """Constructs the FlowJax architecture based on the specified parameters."""
    from flowjax.flows import coupling_flow, masked_autoregressive_flow
    from flowjax.distributions import Transformed, Normal
    import flowjax.bijections as fjb
    import paramax
    
    init_kwargs = init_kwargs or {}
    transformer_kwargs = transformer_kwargs or {'loc': 0, 'scale': 1}
    transformer_cls = transformer_cls or 'Affine'

    # Extract or infer the transformer component
    if 'transformer' in init_kwargs:
        transformer = init_kwargs.pop('transformer')
        if isinstance(transformer, fjb.Affine):
            if not isinstance(transformer.scale, paramax.Parameterize):
                raise ValueError(f"Expected Affine scale to be Parameterize but found {transformer.scale}")
            transformer_cls = 'Affine'
            transformer_kwargs = {'loc': transformer.loc, 'scale': transformer.scale.fn(transformer.scale.args[0])}
        elif isinstance(transformer, fjb.RationalQuadraticSpline):
            transformer_cls = 'RationalQuadraticSpline'
            transformer_kwargs = {'knots': transformer.knots, 'interval': transformer.interval, 'softmax_adjust': transformer.softmax_adjust, 'min_derivative': transformer.min_derivative}
        else:
            raise ValueError("Only Affine and RationalQuadraticSpline are supported as predefined transformers.")

    if 'base_dist' in init_kwargs:
        raise ValueError("base_dist is not supported as an init param in FlowJAXDistribution init_kwargs")

    # Select the base flow function
    if kind == 'coupling':
        init_fn = coupling_flow
    elif kind == 'masked_autoregressive':
        init_fn = masked_autoregressive_flow
    else:
        raise ValueError(f"Unsupported kind '{kind}' passed to initialize the flow.")
    
    # Configure the flow parameters
    transformer = getattr(fjb, transformer_cls)(**transformer_kwargs)
    init_kwargs.setdefault('base_dist', Normal(jnp.zeros(num_params)))
    init_kwargs.setdefault('transformer', transformer)
    init_kwargs.setdefault('nn_width', 50)
    init_kwargs.setdefault('nn_depth', 1)
    init_kwargs.setdefault('flow_layers', 8)
    init_kwargs.setdefault('invert', False)
    
    if init_kwargs['base_dist'].shape[0] != num_params:
        raise ValueError("Base distribution must have shape equal to the number of parameters.")        
    
    # Initialize the base architecture
    flow = init_fn(key, **init_kwargs)                
    
    # Augment the bijection chain to normalize the input data automatically
    loc = (theta_min + theta_max) / 2
    scale = (theta_max - theta_min) / 2
    norm_layer = paramax.NonTrainable(fjb.Affine(loc=loc, scale=scale))
    augmented_bijection = fjb.Chain([flow.bijection, norm_layer])
    
    return Transformed(flow.base_dist, augmented_bijection)


def _train_flow(
    key: jnp.ndarray, 
    flow: Any, 
    samples: jnp.ndarray, 
    weights: jnp.ndarray | None = None, 
    **train_kwargs
) -> Any:
    """Executes the training loop to fit the flow to the provided data."""
    from flowjax.train import fit_to_data
    
    try:
        from paramax import unwrap 
    except ImportError:
        from flowjax.distributions import unwrap

    # Set reasonable defaults for the training loop
    train_kwargs.setdefault('learning_rate', 1e-3)
    train_kwargs.setdefault('batch_size', 64)
    train_kwargs.setdefault('max_patience', 10)

    if weights is not None:
        # Scale weights so they sum to N to prevent vanishing gradients
        n_samples = weights.shape[0]
        sum_weights = weights.sum()
        scale_factor = n_samples / (sum_weights + 1e-10)
        weights = weights * scale_factor

        def weighted_loss_fn(params, static, x, w, key=None):
            dist = unwrap(eqx.combine(params, static))
            return -(w * dist.log_prob(x)).mean()

        data = (samples, weights)
        train_kwargs["loss_fn"] = weighted_loss_fn
    else:
        data = samples

    # Fit the model
    trained_flow, _losses = fit_to_data(key, flow, data, **train_kwargs)
    
    return trained_flow