from typing import Callable

import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.sampling.base import BaseSampler
from pmrf.sampling._algos.field_minimization import FieldSampler
from pmrf.models.model import Model
from pmrf.functions import mean_ax0, l2_norm_ax0, mag_2_db, conv_inter
from pmrf.constants import ArrayFuncT

MAE_COST = [jnp.abs, mean_ax0, mean_ax0, mag_2_db]
CONVOLUTIONAL_COST = [l2_norm_ax0, conv_inter, l2_norm_ax0, mag_2_db]

class SurrogateFieldSampler(FieldSampler):
    """
    Samples new points by minimizing a scalar field induced by a surrogate model.
    
    This is a type of FieldSampler, where the "training" function trains a surrogate model,
    and the "evaluation" function returns the field to minimize as a function of that surrogate.
    
    Convergence is reached when the surrogate approximates the model according to some cost function.
    """
    def __init__(
        self,
        model: Model,
        train_fn: Callable[[jnp.ndarray, jnp.ndarray, Frequency], Model], # params, features, frequency, and `key` as a key-word argument
        eval_fn: Callable[[Model, Frequency], float],
        cost_kind: str | None = None,
        cost_fn: ArrayFuncT | list[ArrayFuncT] | eqx.Module | None = None,
        *args,
        **kwargs
    ):
        if not 'frequency' in kwargs:
            raise Exception("Cannot create a SurrogateSampler without a frequency")
        
        self.surrogate: Model = None
        def train_fn_wrapper(theta: jnp.ndarray, features: jnp.ndarray, frequency: Frequency, key=None) -> Model:
            self.surrogate = train_fn(theta, features, frequency, key=key)
            return self.surrogate
        def eval_fn_wrapper(surrogate: Model, theta: jnp.ndarray, frequency: Frequency, key=None) -> Model:
            surrogate = surrogate.with_params(theta)
            return eval_fn(surrogate, frequency, key=key)

        cost_kind = cost_kind or cost_kind or 'complex'
        features = kwargs.pop('features')
        
        default_features = None
        default_cost = None
        if cost_kind == 'convolutional':
            default_features = ['s', 's_mag']
            default_cost = CONVOLUTIONAL_COST
        elif cost_kind == 'complex':
            default_features = ['s']
            default_cost = MAE_COST
        elif cost_kind == 'magnitude':
            default_features = ['s_mag']
            default_cost = MAE_COST
        elif cost_kind is not None:
            raise Exception("Unknown cost kind alias passed to surrogate sampler")

        if features is None:
            features = default_features
        if cost_fn is None:
            cost_fn = default_cost        
        
        if cost_fn is not None and not isinstance(cost_fn, list):
            cost_fn = [cost_fn]
        cost_fn = cost_fn if isinstance(cost_fn, eqx.Module) else eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost_fn])
        
        self.cost_fn = cost_fn
        self.cost_values = []
        self.surrogate_converged = False
        
        return super().__init__(model=model, features=features, train_fn=train_fn_wrapper, eval_fn=eval_fn_wrapper, *args, **kwargs)
    
    def _generate(self, N: int, d: int, key=None, threshold=None, patience=5, **kwargs) -> jnp.ndarray | None:
        # Return if the surrogate has converged
        if self.surrogate_converged:
            return None
        
        # Generate the samples and train the surrogate
        U_samples = super()._generate(N, d, key, threshold=None, patience=(patience + 1), **kwargs)
        if self.surrogate is None:
            raise Exception("Surrogate cannot be `None` for SurrogateSampler")
        
        # Return if the field has converged
        if U_samples is None:
            return None
        
        # Compute the worst cost for the new samples
        theta_samples = jnp.array([self.inverse_cumulative_distribution_fn(u) for u in U_samples])
        sample_costs = []
        for theta in theta_samples:
            new_actual_features = self.add_sample(theta)
            new_surrogate_features = self.feature_fn(theta, model=self.surrogate)
            error = new_surrogate_features - new_actual_features
            cost = self.cost_fn(error)
            sample_costs.append(cost)
        worst_cost = jnp.max(jnp.array(sample_costs))
        
        # Print and append the worst cost
        self.logger.info(f"Surrogate cost = {worst_cost:.2f}")
        self.cost_values.append(worst_cost)
        
        # Check if we have converged. We only return None for the next round so that these samples still get added
        self.surrogate_converged = self._check_convergence(self.cost_values, threshold, patience)
        return U_samples