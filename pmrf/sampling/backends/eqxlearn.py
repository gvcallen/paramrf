from typing import Callable
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from eqxlearn import BaseModel, fit
from eqxlearn.model_selection import KFold, cross_validate
from eqxlearn.metrics import mean_absolute_error

from pmrf.models import Model
from pmrf.sampling.algorithms import FieldSampler
from pmrf.sampling.base import SampleResults
from pmrf.algorithms.anomaly import get_anomaly_mask

class EqxLearnUncertaintySampler(FieldSampler):
    """
    An adaptive sampler that selects samples to minimize the uncertainty of an 'eqx-learn' surrogate model.
    """
    def __init__(
        self,
        model: Model,
        surrogate: BaseModel,
        cv=None,
        fit_kwargs: dict = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.surrogate = surrogate
        self.cv = cv or partial(KFold, n_splits=5, shuffle=True)
        self.fit_kwargs = fit_kwargs or {}
        self.anomaly_threshold = None
        
    def run(self, anomaly_threshold=1e-3, **kwargs) -> SampleResults:
        self.anomaly_threshold = anomaly_threshold
        return super().run(**kwargs)
    
    def preprocess(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        params, features = self.sampled_params, self.sampled_features
        anomaly_threshold = self.anomaly_threshold
        num_features = features.shape[-1]
        for fi in range(num_features):
            features_i = features[...,fi]
        
            # Filter out NaNs
            mask = jnp.any(jnp.isnan(features))
            mask = mask | jnp.any(jnp.isnan(params))
            
            # Filter out anomalies
            if anomaly_threshold is not None:
                mask = mask | get_anomaly_mask(features_i, anomaly_threshold)
            
            # Apply the masks
            if len(features[mask]) > 0:
                self.logger.warning(f"Not using {len(features[mask])} bad features")
                features = features[~mask]
                if params is not None:
                    params = params[~mask]

        # Flatten features
        features = features.reshape(features.shape[0], -1)
        
        return params, features

    def train_field(self, key=None) -> BaseModel:
        X, y = self.preprocess()
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
    
    def calculate_convergence(self, key=None) -> float:
        X, y = self.preprocess()
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