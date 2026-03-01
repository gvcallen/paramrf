from typing import Callable
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from eqxlearn import BaseModel, fit
from eqxlearn.model_selection import KFold, cross_validate
from eqxlearn.metrics import mean_absolute_error

from pmrf.models import Model
from pmrf.sampling._algorithms import FieldSampler

class EqxLearnSurrogateSampler(FieldSampler):
    """
    An adaptive sampler that selects samples to minimize the uncertainty of an 'eqx-learn' surrogate model.

    The eqx-learn surrogate can be any model that is capable of returning its prediction variance using `return_var=True`.
    Convergence occurs when the cross_validation metric converges. By default, this is KFold with 5 splits.
    
    Note that the features are flattened for each sample before training the model. Specifically, if the feature matrix
    for one sample is of shape `(nfreq, nfeatures)`, then a matrix of shape (nfreq x nfeatures,) is passed.
    Note that this occurs AFTER any pre-process function is called.
    """
    def __init__(
        self,
        model: Model,
        surrogate: BaseModel,
        preprocess_fn: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] = None, # theta, features
        cv=None,
        fit_kwargs: dict = None,
        *args,
        **kwargs
    ):
        # self 
        
        def preprocess(theta, features):
            if preprocess_fn is not None:
                theta, features = preprocess_fn(theta, features)
            features = features.reshape(features.shape[0], -1)
            return theta, features

        if cv is None:
            cv = partial(KFold, n_splits=5, shuffle=True)

        # Setup surrogate model
        D = model.num_flat_params

        # Define the training callback
        def train(theta: jnp.ndarray, features: jnp.ndarray, key=None) -> BaseModel:    
            X, y = preprocess(theta, features)
            self.logger.info(f"Training surrogate model...")
            fitted_eqx_model, losses = fit(surrogate, X=X, y=y, **fit_kwargs, key=key)
            self.logger.info(f"Final loss: {losses[-1]:.2f}")
            return fitted_eqx_model

        # Define the variance callback
        def variance(model: BaseModel, theta: jnp.ndarray, key=None) -> float:
            _y_mean, y_var = model(theta, return_var=True)
            # pca_noise = model.func['regressor'].transformer['y_pca'].noise_variance
            rayleigh_factor = jnp.sqrt(jnp.pi) / 2.0
            total_std = jnp.sqrt(y_var.real + y_var.imag)
            expected_mae = jnp.mean(rayleigh_factor * total_std)
            return 20*jnp.log10(expected_mae)
        
        # Define the validation callback for convergence
        def validate(theta: jnp.ndarray, features: jnp.ndarray, key=None) -> float:
            nonlocal cv
            X, y = preprocess(theta, features)
            key, split_key = jr.split(key)
            cv_instance = cv(key=split_key)
            mae_db = lambda model, X, y: jnp.max(20*jnp.log10(mean_absolute_error(y, model.predict(X), axis=1)))
            
            # Run the validation
            self.logger.info(f"Validating surrogate model...")
            key, cv_key = jr.split(key)
            results = cross_validate(surrogate, X, y, cv=cv_instance, scoring=mae_db, return_loss=True, key=cv_key, **fit_kwargs)
            score, losses = jnp.mean(results['test_score']), results['loss']
            self.logger.info(f"Average error = {score:.2f} dB. Traing losses = {[round(float(loss), 2) for loss in losses]}")
            return score
        
        return super().__init__(model=model, train_fn=train, eval_fn=variance, convergence_fn=validate, *args, **kwargs)