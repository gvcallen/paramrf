from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import clamp_probs
import jax.numpy as jnp
import jax
from jax.random import PRNGKey
from typing import Any

from pmrf.parameters import ParameterGroup

class JointParameterDistribution(Distribution):
    support = constraints.real_vector
    reparametrized_params = []
    has_rsample = True

    def __init__(self, param_groups: list[ParameterGroup], param_names: list[str], kind='prior'):
        self.param_groups = param_groups
        self.param_names = param_names
        self.kind = kind
        self.name_to_index = {name: i for i, name in enumerate(param_names)}
        self._validate_groups()

        event_shape = (len(param_names),)
        batch_shape = ()
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def _validate_groups(self):
        for group in self.param_groups:
            if not hasattr(group, self.kind):
                raise ValueError(f"Parameter group {group} has no {self.kind} defined.")
            dist: Distribution = getattr(group, self.kind)
            if not hasattr(dist, "sample") or not hasattr(dist, "log_prob"):
                raise ValueError(f"Parameter group prior must support sample and log_prob.")

    def sample(self, key, sample_shape=()):
        values = [None] * len(self.param_names)
        for group in self.param_groups:
            group_names = group.param_names
            subkey, key = jax.random.split(key)
            dist: Distribution = getattr(group, self.kind)
            samples = dist.sample(subkey, sample_shape)
            for i, name in enumerate(group_names):
                idx = self.name_to_index[name]
                values[idx] = samples[..., i]
        return jnp.stack(values, axis=-1)

    def log_prob(self, value):
        total_logp = 0.0
        for group in self.param_groups:
            group_names = group.param_names
            idxs = [self.name_to_index[name] for name in group_names]
            group_vals = value[..., idxs]
            dist: Distribution = getattr(group, self.kind)
            logp = dist.log_prob(group_vals)
            total_logp += logp
        return total_logp

    def icdf(self, u):
        values = [None] * len(self.param_names)
        for group in self.param_groups:
            group_names = group.param_names
            idxs = [self.name_to_index[name] for name in group_names]
            group_u = u[..., idxs]
            dist: Distribution = getattr(group, self.kind)
            group_x = dist.icdf(group_u)
            for i, name in enumerate(group_names):
                values[self.name_to_index[name]] = group_x[..., i]
        return jnp.stack(values, axis=-1).reshape(u.shape)

    def cdf(self, x):
        values = [None] * len(self.param_names)
        for group in self.param_groups:
            group_names = group.param_names
            idxs = [self.name_to_index[name] for name in group_names]
            group_x = x[..., idxs]
            dist: Distribution = getattr(group, self.kind)
            group_u = dist.cdf(group_x)
            for i, name in enumerate(group_names):
                values[self.name_to_index[name]] = group_u[..., i]
        return jnp.stack(values, axis=-1)

    def support(self):
        return constraints.real_vector  # Override if more specific support is known