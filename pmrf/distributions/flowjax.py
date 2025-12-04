import jax
import jax.numpy as jnp
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import pickle

from numpyro.distributions import Distribution

from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Transformed, Normal

class FlowJAXDistribution(Distribution):
    """
    JAX-native Adapter for models using FlowJax. Currently only supports MAF.
    """
    theta_min: jnp.ndarray
    theta_max: jnp.ndarray
    flow = None
    # event_dim: int

    def __init__(self, flow: Transformed, theta_min=None, theta_max=None, validate_args=None):
        self.flow = flow
        event_dim = flow.shape[0]
        event_shape = (event_dim,)
        self.theta_min = theta_min if theta_min is not None else jnp.zeros(event_dim)
        self.theta_max = theta_max if theta_max is not None else jnp.ones(event_dim)

        super().__init__(batch_shape=(), event_shape=event_shape, validate_args=validate_args)    

    @classmethod
    def generate(cls, data, flow_kwargs: dict = {}, learning_rate=1e-3, epochs=100, kind='maf', **fit_kwargs):
        key = jax.random.PRNGKey(42)

        data = jnp.asarray(data)
        theta_min = jnp.min(data, axis=0)
        theta_max = jnp.max(data, axis=0)
        flow_key, train_key = jax.random.split(key)

        # self.number_networks = kwargs.pop('number_networks', 6)
        # self.hidden_layers = kwargs.pop('hidden_layers', [50, 50])        

        if kind != 'maf':
            raise Exception(f'Currently only {kind} distributions are supported for FlowJAXDistribution')
        
        flow = masked_autoregressive_flow(
            key=flow_key,
            base_dist=Normal(loc=jnp.zeros(data.shape[1])),
            flow_layers=6,
            nn_width=50,
            nn_depth=2,
            nn_activation=jax.nn.tanh,
            **flow_kwargs
        )

        flow, losses = fit_to_data(
            key=train_key,
            dist=flow,
            data=data,
            learning_rate=learning_rate,
            max_epochs=epochs,
            **fit_kwargs,
        )

        return FlowJAXDistribution(flow, theta_min, theta_max)

    def sample(self, key, sample_shape=()):
        # TODO remove when old-style keys fully deprecated
        if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jr.wrap_key_data(key)

        return self.flow.sample(key, sample_shape)

    def log_prob(self, value):
        return self.flow.log_prob(value)

    def icdf(self, u):
        u = jnp.atleast_2d(u)
        
        CLIP = 1e-9
        u = jnp.clip(u, CLIP, 1.0 - CLIP)
        z = jax.scipy.stats.norm.ppf(u)
        
        return jax.vmap(self.flow.bijection.transform)(z)

    def save(self, path: str):
        with open(path, "wb") as f:
            eqx.tree_serialise_leaves(f, self)
            pickle.dump({'min': self.theta_min, 'max': self.theta_max}, f)

    @staticmethod
    def load(path: str, flow_structure_proxy=None):
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                raise NotImplementedError(
                    "Requires a model skeleton. "
                    "Use standard pickle for full object persistence or instantiate "
                    "a skeleton FlowJaxMAFAdapter and use eqx.tree_deserialise_leaves."
                )

    @property
    def min(self):
        return self.theta_min

    @property
    def max(self):
        return self.theta_max