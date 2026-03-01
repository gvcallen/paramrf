from typing import Callable
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from eqxlearn import BaseModel, fit
from eqxlearn.model_selection import KFold, cross_validate
from eqxlearn.metrics import mean_absolute_error

from pmrf.models import Model
from pmrf.sampling.algorithms import FieldSampler

class EqxLearnSurrogateSampler(FieldSampler):
    """
    An adaptive sampler that selects samples to minimize the uncertainty of an 'eqx-learn' surrogate model.
    """
    def __init__(
        self,
        model: Model,
        surrogate: BaseModel,
        preprocess_fn: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] = None,
        cv=None,
        fit_kwargs: dict = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.surrogate = surrogate
        self.preprocess_fn = preprocess_fn
        self.cv = cv or partial(KFold, n_splits=5, shuffle=True)
        self.fit_kwargs = fit_kwargs or {}

    def _preprocess(self, theta: jnp.ndarray, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.preprocess_fn is not None:
            theta, features = self.preprocess_fn(theta, features)
        features = features.reshape(features.shape[0], -1)
        return theta, features

    def train_field(self, params: jnp.ndarray, features: jnp.ndarray, key=None) -> BaseModel:    
        X, y = self._preprocess(params, features)
        self.logger.info("Training surrogate model...")
        fitted_eqx_model, losses = fit(self.surrogate, X=X, y=y, key=key, **self.fit_kwargs)
        self.logger.info(f"Final loss: {losses[-1]:.2f}")
        return fitted_eqx_model

    def evaluate_field(self, field: BaseModel, theta: jnp.ndarray, key=None) -> float:
        _y_mean, y_var = field(theta, return_var=True)
        rayleigh_factor = jnp.sqrt(jnp.pi) / 2.0
        total_std = jnp.sqrt(y_var.real + y_var.imag)
        expected_mae = jnp.mean(rayleigh_factor * total_std)
        return 20 * jnp.log10(expected_mae)
    
    def calculate_convergence(self, params: jnp.ndarray, features: jnp.ndarray, key=None) -> float:
        X, y = self._preprocess(params, features)
        key, split_key = jr.split(key)
        cv_instance = self.cv(key=split_key)
        
        def mae_db(model, X, y):
            return jnp.max(20 * jnp.log10(mean_absolute_error(y, model.predict(X), axis=1)))
        
        self.logger.info("Validating surrogate model...")
        key, cv_key = jr.split(key)
        results = cross_validate(
            self.surrogate, X, y, 
            cv=cv_instance, scoring=mae_db, return_loss=True, key=cv_key, **self.fit_kwargs
        )
        
        score = jnp.mean(results['test_score'])
        losses = results['loss']
        loss_str = [round(float(loss), 2) for loss in losses]
        self.logger.info(f"Average error = {score:.2f} dB. Training losses = {loss_str}")
        return score