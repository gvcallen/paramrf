from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.sampling._algorithms.field import FieldSampler
from pmrf.models.model import Model
from pmrf.math_functions import mean_ax0, mag_2_db
from pmrf.constants import ArrayFuncT

MEAN_ABSOLUTE_ERROR = [jnp.abs, mean_ax0, mean_ax0, mag_2_db]
ROOT_MEAN_SQUARED_ERROR = [jnp.abs, jnp.sqrt, mean_ax0, mean_ax0, lambda x: x**2, mag_2_db]

class SurrogateFieldSampler(FieldSampler):
    """
    Samples new points at the maxima of a scalar field induced by a surrogate model.
    
    This is a type of FieldSampler, where the "training" function trains a surrogate model,
    and the "evaluation" function returns the field to minimize as a function of that surrogate.
    
    Convergence is reached when the surrogate approximates the model according to some cost function,
    AND when the field maxima stop decreasing.
    """
    def __init__(
        self,
        model: Model,
        train_fn: Callable[[jnp.ndarray, jnp.ndarray], Model], # params, features, and `key=key`
        eval_fn: Callable[[Model], float], # model and `key=key`
        validate_fn: Callable[[jnp.ndarray, jnp.ndarray], float], # params, features, and `key=key`
        error_kind: str | None = None,
        error_fn: ArrayFuncT | list[ArrayFuncT] | eqx.Module | None = None,
        *args,
        **kwargs
    ):
        if not 'frequency' in kwargs:
            raise Exception("Cannot create a SurrogateSampler without a frequency")
        
        def eval_fn_wrapper(surrogate: Model, theta: jnp.ndarray, frequency: Frequency, key=None) -> Model:
            surrogate = surrogate.with_params(theta)
            return eval_fn(surrogate, frequency, key=key)
        
        error_kind = error_kind or error_kind or 'complex'
        features = kwargs.pop('features')
        
        default_features = None
        default_error = None
        if error_kind == 'complex':
            default_features = ['s']
            default_error = MEAN_ABSOLUTE_ERROR
        elif error_kind == 'magnitude':
            default_features = ['s_mag']
            default_error = MEAN_ABSOLUTE_ERROR
        elif error_kind is not None:
            raise Exception("Unknown error kind alias passed to surrogate sampler")

        if features is None:
            features = default_features
        if error_fn is None:
            error_fn = default_error        
        
        if error_fn is not None and not isinstance(error_fn, list):
            error_fn = [error_fn]
        error_fn = error_fn if isinstance(error_fn, eqx.Module) else eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in error_fn])
        
        self.error_fn = error_fn
        self.error_values = []
        self.surrogate_converged = False
        
        return super().__init__(model=model, features=features, train_fn=train_fn, eval_fn=eval_fn_wrapper, convergence_fn=validate_fn, *args, **kwargs)
    
    def _generate(self, N: int, d: int, *, key=None, threshold=None, patience=5, **kwargs) -> jnp.ndarray | None:
        surrogate = self.field
        
        # Generate the samples and train the surrogate
        key, generate_key = jr.split(key)
        U_samples = super()._generate(N, d, key=generate_key, threshold=threshold, patience=patience, **kwargs)
        if surrogate is None:
            raise Exception("Surrogate cannot be `None` for SurrogateSampler")
        
        # Return if the sampling has converged.
        if U_samples is None:
            return None
        
        # Compute the worst cost for the new samples
        theta_samples = jnp.array([self.inverse_cumulative_distribution_fn(u) for u in U_samples])
        sample_errors = []
        new_actual_features_all = self.add_samples(theta_samples)
        for i, theta in enumerate(theta_samples):
            new_actual_features = new_actual_features_all[i]
            new_surrogate_features = self.feature_fn(theta, model=surrogate)
            error = new_surrogate_features - new_actual_features
            error = self.error_fn(error)
            sample_errors.append(error)
        
        # Print and append the worst cost
        worst_error = jnp.max(jnp.array(sample_errors))
        self.error_values.append(worst_error)
        if N == 1:
            self.logger.info(f"Surrogate error = {worst_error:.2f}")
        else:
            error_str = ""
            for error in sample_errors:
                error_str += f"{error:.2f}, "
            error_str = error_str[0:len(error_str)-2]
            self.logger.info(f"Surrogate errors = [{error_str}]")
            
        return U_samples