from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.explore.field import AbstractFieldSampler

class EqxLearnUncertaintySampler(AbstractFieldSampler):
    """Adaptive sampler targeting regions of high surrogate uncertainty."""
    surrogate: Any = None
    fit_kwargs: dict = eqx.field(default_factory=dict)

    def train_field(self, params: jnp.ndarray, features: jnp.ndarray, key: jax.Array) -> Any:
        from eqxlearn import fit
        
        # Flatten features for eqx-learn
        flat_features = features.reshape(features.shape[0], -1)
        fitted_model, _ = fit(self.surrogate, X=params, y=flat_features, key=key, **self.fit_kwargs)
        return fitted_model

    def evaluate_field(self, field: Any, theta: jnp.ndarray, key: jax.Array) -> float:
        _y_mean, y_var = field(theta, return_var=True)
        rayleigh_factor = jnp.sqrt(jnp.pi) / 2.0
        total_std = jnp.sqrt(y_var.real + y_var.imag)
        expected_mae = jnp.mean(rayleigh_factor * total_std)
        return 20 * jnp.log10(expected_mae)