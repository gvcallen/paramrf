import numpy as np
import skrf
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.functions import l2_norm_ax0, mag_2_db
from pmrf._model import Model
from pmrf.parameters import Parameter, ParameterGroup
from pmrf._constants import FeatureInputT, ArrayFuncT

from pmrf.fitting._base import BaseFitter, FitResults

class FrequentistResults(FitResults):
    pass

class FrequentistFitter(BaseFitter):
    """
    **Overview**

    A base class for frequentist (classical) optimization methods.

    This class extends `BaseFitter` by adding the concept of a `cost_fn`,
    a function that takes the difference between model features and measured
    features and computes a single scalar value representing the "cost" or "error".
    The goal of the fitter is to minimize this value.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | dict[str, skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureInputT | None = None,
        cost: ArrayFuncT | list[ArrayFuncT] | eqx.Module = None,
        *args, **kwargs
    ) -> None:
        """Initializes the FrequentistFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):
                The measured network data to fit the model against.
            frequency (skrf.Frequency | None, optional):
                The frequency axis to perform the fit on. Defaults to `None`.
            features (FeatureT | FeatureListT | None, optional),
                The features to extract for comparison. Defaults to `None`.
            cost (ArrayFuncT | list[ArrayFuncT] | eqx.Module, optional):
                A function or sequence of functions defining the cost metric. If a list
                of functions is provided, they are composed sequentially. If `None`, a
                default cost function (typically L2 norm on the dB magnitude difference)
                is used. Defaults to `None`.
        """
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        if cost is not None and not isinstance(cost, list):
            cost = [cost]
        if cost is None:
            if len(features) > 1:
                cost = [mag_2_db, l2_norm_ax0, l2_norm_ax0]
            else:
                cost = [mag_2_db, l2_norm_ax0]
        
        self.cost_metric_fn = cost if isinstance(cost, eqx.Module) else eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost])
        
    def _make_cost_function(self, as_numpy=False):
        x0_jax = self.initial_model.flat_params()
        feature_fn_jax = self._make_feature_function()

        # Define the JAX cost function to be minimized
        @jax.jit
        def cost_fn(flat_params) -> jnp.ndarray:
            model_features = feature_fn_jax(flat_params)
            error = self.measured_features - model_features
            cost_val = self.cost_metric_fn(error)
            if jnp.isscalar(self.cost_metric_fn(error)):
                return cost_val
            else:
                return cost_val[0]
            
        if as_numpy:
            cost_fn_jax = cost_fn
            cost_fn = lambda x: float(cost_fn_jax(jnp.array(x)))
            x0_np = np.array(x0_jax)
            x0 = x0_np
            
        self.logger.info(f"Compiling cost function...")
        _c0 = cost_fn(x0)
        
        return cost_fn
    
    def _bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        param_groups = self.initial_model.param_groups()
        param_names = self.initial_model.flat_param_names()
        
        name_to_minimum = {name: None for name in param_names}
        name_to_maximum = {name: None for name in param_names}
        
        for param_group in param_groups:
            group_minimums, group_maximums = param_group.min, param_group.max
            group_param_names = list(param_group.params.keys())
            for i, name in enumerate(group_param_names):
                name_to_minimum[name] = group_minimums[i]
                name_to_maximum[name] = group_maximums[i]
        
        if any(value is None for value in name_to_minimum.values()) or any(value is None for value in name_to_maximum.values()):
            raise Exception('Parameter found that did not belong to a parameter groups')
        
        return jnp.array(list(name_to_minimum.values())), jnp.array(list(name_to_maximum.values()))