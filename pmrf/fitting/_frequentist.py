import numpy as np
import skrf
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.functions import l2_norm_ax0, mag_2_db
from pmrf._model import Model
from pmrf._constants import FeatureT, FeatureListT, ArrayFuncT

from pmrf.fitting._base import BaseFitter, FitResults
from pmrf.fitting._features import make_feature_function

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
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureT | FeatureListT | None = None,
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
        
    def _make_cost_fn(self, dont_jit=False):
        # Generate JAX-compatible functions for feature extraction and model reconstruction
        feature_fn, x0, recon_fn = make_feature_function(self.model, self.feature_list, self.model_frequency, flat=True)
        self._cached_numpy_cost = feature_fn, x0, recon_fn

        # Define the JAX cost function to be minimized
        def cost_jax(flat_params) -> jnp.ndarray:
            model_features = feature_fn(flat_params)
            error = self.measured_features - model_features
            cost_val = self.cost_metric_fn(error)
            if jnp.isscalar(self.cost_metric_fn(error)):
                return cost_val
            else:
                return cost_val[0]

        if not dont_jit:
            self.logger.info("Compiling model and cost function...")
            cost_jax = jax.jit(cost_jax)
            _ = cost_jax(x0)
        
        cost_numpy = lambda x: float(cost_jax(jnp.array(x)))
            
        return cost_numpy, recon_fn, np.array(x0)