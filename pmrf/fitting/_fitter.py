from abc import ABC, abstractmethod
from typing import Callable

import scipy
import scipy.optimize
import skrf
import jax
import optax
import equinox as eqx

from pmrf.model import Model
from pmrf.parameter import Parameter, fixed
from pmrf.system import ModelSystem
from pmrf.fitting._results import BayesianResults, FrequentistResults
from pmrf.frequency import Frequency
from pmrf._numpy import USE_JAX
from pmrf._numpy import numpy as np

from pmrf.fitting._feature import Feature, features_from_strings, extract_features
from pmrf.fitting._modifier import Modifier, modifiers_from_strings, apply_modifiers

from dataclasses import dataclass

import skrf

from pmrf.frequency import Frequency
from pmrf.model import Model
from pmrf.system import ModelSystem
from pmrf._numpy import numpy as np


# def Fitter(
#     model: Model | list[Model] | ModelSystem,
#     measured: skrf.Network | list[skrf.Network],
#     params: dict[Parameter] = None,
#     fit_frequency: skrf.Network = None,
#     engine: str = 'scipy',
#     solver: str | None = 'BFGS',
#     features: str | FeatureExtractorSet = None,
#     cost_steps: str | ModifierChain = None,
#     max_iterations: int | None = None,    
#     **kwargs
# ) -> 'BaseFitter':
#     if engine == 'scipy':
#         return FrequentistFitter()


class BaseFitter(ABC):
    def __init__(
        self,
        model: Model | ModelSystem,
        measured: skrf.Network | list[skrf.Network],
        params: dict[str, Parameter] | None = None,
        fit_frequency: skrf.Network | None = None,
        features: list[Feature] | list[str] = None,
        param_infix = '_'
    ) -> None:
        if features is None:
            if isinstance(model, Model):
                features = [Feature(mode='complex', property='s', ports=(0, 0), scale='lin')]
            else:
                features = [Feature(mode='complex', property='s', ports=(0, 0), scale='lin'), Feature(mode='magnitude', property='s', ports=(0, 0), scale='lin')]
        
        if isinstance(features[0], str):
            features = features_from_strings(features)

        if isinstance(measured, list) and len(measured) > 1:
            if fit_frequency is None:
                raise ValueError("Error: Currently `fit_frequency` must be passed for multi-measurement fits (i.e. all networks must be explicitly interpolated onto the same frequency) for fitting")
        
            measured_interp = []
            for ntwk in measured:
                measured_interp.append(ntwk.interpolate(fit_frequency))
            measured = measured_interp
        
        self.model: Model | ModelSystem = model
        self.measured: skrf.Network | list[skrf.Network] = measured
        self.params: dict[str, Parameter] = params or {}
        self.param_infix = param_infix
        self.features = features
        self.measured_features = extract_features(self.measured, self.features)

        self._init_params()

    def _init_params(self):
        # The most important part - making sure our model params are in the same order as the model's
        user_params = self.params
        model_params = self.model.params(separator=self.param_infix)
        final_params = {}
        for name, value in model_params.items():
            # TODO add more complicated parameter initialization options e.g. default % width.
            # Also add error check if the user passes a parameter that is NOT in the model
            final_params[name] = user_params[name] if name in user_params else fixed(value)

        self.params = final_params

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @property
    def param_names(self) -> list[str]:
        return list(self.params.keys())
    
    @property
    def param_names_free(self) -> list[str]:
        return list(k for k, v in self.params.items() if not v.fixed)
    
    @property
    def param_names_fixed(self) -> list[str]:
        return list(k for k, v in self.params.items() if v.fixed)
    
    @property
    def param_values(self) -> np.ndarray:
        return np.array([v.value for v in self.params.values()])
    
    @property
    def param_values_free(self) -> np.ndarray:
        return np.array([v.value for v in self.params.values() if not v.fixed])
    
    @property
    def param_values_fixed(self) -> np.ndarray:
        return np.array([v.value for v in self.params.values() if v.fixed])
    
    @property
    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([v.lower for v in self.params.values()])
        upper = np.array([v.upper for v in self.params.values()])
        return lower, upper
    
    @property
    def param_bounds_free(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([v.lower for v in self.params.values() if not v.fixed])
        upper = np.array([v.upper for v in self.params.values() if not v.fixed])
        return lower, upper    
    
    @property
    def param_bounds_fixed(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([v.lower for v in self.params.values() if v.fixed])
        upper = np.array([v.upper for v in self.params.values() if v.fixed])
        return lower, upper


class FrequentistFitter(BaseFitter):
    def __init__(self, modifiers: list[Modifier] | list[str] = None, *args, **kwargs):
        BaseFitter.__init__(self, *args, **kwargs)
        
        if modifiers is None:
            modifiers = ['L2', 'convolve-interleaved', 'L2', 'dB']

        if isinstance(modifiers[0], str):
            modifiers = modifiers_from_strings(modifiers)
        
        self.modifiers = modifiers

    def cost(self, model: Model | ModelSystem | None = None) -> np.ndarray:
        """Returns the cost for the model and the specified measured data.

        The cost is calculated by first extracting "feature" matrix from the model (such as S11 magnitude) using the `FeatureExtractor` objects in `self.features`,
        and then applying "modifiers" (such as by taking the L2 norm) on the resultant matrix using the `Modifier` objects in `self.modifiers`.
        See `Feature` and `Modifier` for more details.
        
        Returns:
            `np.ndarray`: The cost function value.
        """
        model = model or self.model
        model_features = extract_features(model, self.features, self.measured.frequency)
        measured_features = self.measured_features
        
        residuals = measured_features - model_features
        return apply_modifiers(residuals, self.modifiers)
    
    @property
    def param_cost_function(self) -> Callable[[np.ndarray], float]:
        @jax.jit
        def cost_fn(theta, *args, **kwargs):
            self.model = self.model.with_params(flat_params=theta)
            cost = self.cost(self.model)
            return cost
        return cost_fn
    
    @property
    def model_cost_function(self) -> Callable[[Model | ModelSystem], float]:
        @jax.jit
        def cost_fn(model, *args, **kwargs):
            cost = self.cost(model)
            return cost
        return cost_fn


class ScipyFitter(FrequentistFitter):
    def run(self,
            *args,
            **kwargs
        ):
        # Populate bounds and options
        x0 = self.param_values_free
        minimums, maximums = self.param_bounds_free
        bounds = scipy.optimize.Bounds(minimums, maximums)
        
        options = {
            'maxiter': 10000
        }

        # Setup the cost function lambda to pass to scipty
        callback_args = {
            'i_solver': 0,
        }

        def progress_callback(xk):            
            callback_args['i_solver'] += 1
            i = callback_args['i_solver']
            if i % 10 == 0:
                cost = self.param_cost_function(xk)
                print(f'{i} = {cost}')

        # Run the minization routine
        result = scipy.optimize.minimize(self.param_cost_function, x0, args=callback_args, bounds=bounds, method='Nelder-Mead', options=options, callback=progress_callback)

        self.model = self.model.with_params(result.x)
        return result


class OptaxFitter(FrequentistFitter):
    def run(self, 
            num_steps: int = 2000, 
            learning_rate: float = 0.001, 
            *args, **kwargs):
        
        optim = optax.adam(learning_rate)
        model = self.model

        # Partition the model to get the initial tree of fittable parameters
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        # loss = self.model_cost_function
        loss = lambda model: self.cost(model)
        print(f'starting loss: {loss(self.model)}')

        @eqx.filter_jit
        def make_step(
            model: Model,
            opt_state,
        ):
            loss_value, grads = eqx.filter_value_and_grad(loss)(model)
            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        print("Starting JAX-native optimization with Optax...")
        for i in range(num_steps):
            model, opt_state, train_loss = make_step(model, opt_state)
            if i % 200 == 0:
                print(f"Step {i}, Loss: {train_loss:.6f}")
        
        print("Optimization finished!")

        # You can create a result object similar to SciPy's
        result = {
            "x": list(model.params().values()),
            "fun": loss,
            "success": True,
            "message": "Optimization terminated successfully.",
        }

        self.model = model
        return result    