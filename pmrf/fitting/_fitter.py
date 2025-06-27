from abc import ABC, abstractmethod

import logging
import skrf
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.functions import l2_norm_ax0, mag_2_db
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._constants import FeatureListT, ArrayFuncT
from pmrf._tree import combine

from pmrf.fitting._features import extract_features, generate_feature_function
from pmrf.fitting._results import FitResults, FrequentistResults, BayesianResults

class BaseFitter(ABC):
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureListT = None,
    ) -> None:
        """The base fitter initializer.

        Args:
            model (Model):                                              The model to fit.
            measured (skrf.Network | list[skrf.Network]):               The measured networks to fit against. If a list is passed, 
                                                                        the networks are viewed as being part of a large, stacked N-port network.
                                                                        If a measurement is not available, an empty network can be passed.
            frequency (skrf.Frequency | None, optional):                The frequency to fit at. Defaults to `None`, in which case
                                                                        the measured frequencies are used (which must be equal).
            features (FeatureListT, optional):                          The features to extract from the models and networks for cost functions, likelihoods etc.
                                                                        Each string is a function or property of the model or network respectively
                                                                        (e.g. 's_db', 's_mag' etc.), and `ports` are the ports to use as a tuple (e.g. (0, 0)).
                                                                        If a list of strings is passed, the features are extracted for each port
                                                                        within each network/model and stacked column-wise into a "feature matrix".
                                                                        If a list of strings-tuple pairs are passed, then each feature is extracted
                                                                        for each port individually, where port numbers are for the full model
                                                                        (e.g. the stacked network in the case where a list of measurements are passed).
                                                                        Defaults to `None`, in which case S11 is used.
        """
        features = features or [('s', (0, 0))]
        
        # All frequencies must be the same across all measurements (at least currently..)
        measured = [measured] if not isinstance(measured, list) else measured
        if frequency is not None:
            measured = [ntwk.interpolate(frequency) for ntwk in measured]
            measured_freq = frequency
        else:
            measured_freq = measured[0].frequency
            for ntwk in measured:
                if ntwk.frequency != measured_freq and not len(ntwk.frequency) == 0:
                    raise ValueError("Error: Currently `fit_frequency` must be passed for multi-measurement fits (i.e. all networks must be explicitly interpolated onto the same frequency for fitting)")
                
        # Initialize model parameters from user and store in flat array
        self.model: Model = model
        self.model_frequency = Frequency.from_skrf(measured_freq)
        self.measured: list[skrf.Network] = measured
        self.measured_frequency = measured_freq
        self.measured_features = extract_features(measured, features)
        self.feature_list = features
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def run(self, *args, **kwargs) -> FitResults:
        pass
    
class FrequentistFitter(BaseFitter):
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureListT | None = None,
        cost: ArrayFuncT | list[ArrayFuncT] | eqx.Module = None,
        *args, **kwargs
    ) -> None:
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        if cost is not None and not isinstance(cost, list):
            cost = [cost]
        if cost is None:
            if len(features) > 1:
                cost = [mag_2_db, l2_norm_ax0, l2_norm_ax0]
            else:
                cost = [mag_2_db, l2_norm_ax0]
        self.cost_fn = cost if isinstance(cost, eqx.Module) else eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost])

class ScipyMinimizeFitter(FrequentistFitter):
    def run(self, *args, **kwargs):
        from scipy.optimize import minimize, Bounds
        
        params_list = self.model.to_params_list()
        minimums = [p.min for p in params_list]
        maximums = [p.max for p in params_list]
        bounds = Bounds(minimums, maximums)
        feature_fn, x0, recon_fn = generate_feature_function(self.model, self.feature_list, self.model_frequency, flat=True)

        def cost_jax(flat_params) -> jnp.ndarray:
            x = self.measured_features - feature_fn(flat_params)
            return self.cost_fn(x)

        self.logger.info("Compiling model..")
        cost_jit = jax.jit(cost_jax)
        def cost_scipy(x, callback_args):
            flat_params_jax = jnp.array(x)
            cost = float(cost_jit(flat_params_jax))
            i = callback_args['fevel']
            if i % 500 == 0:
                self.logger.info(f"fevel = {i}, cost = {cost:.2f}")
            callback_args['fevel'] = i + 1
            return cost

        callback_args = {'fevel': 0}
        scipy_result = minimize(cost_scipy, x0, args=callback_args, bounds=bounds, *args, **kwargs)
        model_opt = recon_fn(scipy_result.x)
        return FrequentistResults(model=model_opt, engine_results=scipy_result)

