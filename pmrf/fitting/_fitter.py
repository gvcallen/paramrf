from functools import partial
from abc import ABC, abstractmethod
from typing import Callable

import jax.scipy.optimize
import scipy
import scipy.optimize
import skrf
import jax
import optax
import equinox as eqx
import jaxopt

import pmrf.numpy as np
from pmrf._model import Model
from pmrf.parameters import Parameter, fixed
from pmrf.fitting._results import BayesianResults, FrequentistResults
from pmrf._frequency import Frequency
from pmrf.numpy import USE_JAX

from pmrf.fitting._feature import Feature, extract_features
from pmrf.functions import mag_2_db, convolve_interleaved

from dataclasses import dataclass

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
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        params: dict[str, Parameter] | None = None,
        fit_frequency: skrf.Frequency | None = None,
        features: list[Feature] = None,
        param_infix = '_'
    ) -> None:
        """The base fitter initialization.

        Args:
            model (Model): The model to fit.
            measured (skrf.Network | list[skrf.Network]): The measured networks to fit against. If a list is passed, the networks are viewed as being part of a large N-port network, with ports sequentially assigned. If a measurement is not available, an empty network can be passed. See `SystemModel` for an example use-case.
            params (dict[str, Parameter] | None, optional): Parameters for the models, specified in a flattened format. See `param_infix`. Defaults to `None`, in which case all parameters are set as normal with 5% standard deviation.
            fit_frequency (skrf.Frequency | None, optional): The frequency to fit against. Defaults to `None`.
            features (list[Feature], optional): The "features" to extract from the networks for cost functions, likelihoods etc. e.g. S11 magnitude. Defaults to `None`, in which case all complex reflection coefficients across all ports are used.
            param_infix (str, optional): The infix between submodels for the flattened parameters in `params`. Parameters are specified as `{model.name}{infix}{submodel1.name}{infix}{submodel2.name}{...}{infix}{param}`. Defaults to '_'.
        """
        # By default, we setup the features to extract the complex reflection coefficients
        if features is None:
            features = []
            for i in range(model.nports):
                features.append(Feature(mode='complex', property='s', ports=(i, i), scale='lin'))
        
        # Currently, all frequencies must be the same across all measurements
        measured = [measured] if not isinstance(measured, list) else measured
        if fit_frequency is not None:
            measured = [ntwk.interpolate(fit_frequency) for ntwk in measured]
        else:
            freq = measured[0].frequency
            for ntwk in measured:
                if ntwk.frequency != freq and not len(ntwk.frequency) == 0:
                    raise ValueError("Error: Currently `fit_frequency` must be passed for multi-measurement fits (i.e. all networks must be explicitly interpolated onto the same frequency for fitting)")
        
        self.model: Model = model
        self.measured: list[skrf.Network] = measured
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
    def __init__(self, cost_fn: list[Callable[[np.ndarray], np.ndarray]] | eqx.Module = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if cost_fn is None:
            L2 = partial(np.linalg.norm, order=2)
            cost_fn = [L2, partial(convolve_interleaved, axis=1), L2, mag_2_db]

        self.cost_fn = eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost_fn])

    def cost(self, model: Model | None = None) -> np.ndarray:
        """Returns the cost for the model and the specified measured data.

        The cost is calculated by first extracting the "feature" matrix from the model (such as S11 magnitude)
        using the `FeatureExtractor` objects in `self.features`. Then, "modifiers" (such as taking the L2 norm)
        are applied sequentially on the resultant matrix, using the `Modifier` objects in `self.modifiers`.
        
        See `Feature` and `Modifier` for more details.
        
        Returns:
            `np.ndarray`: The cost function value.
        """
        model = model or self.model
        model_features = extract_features(model, self.features, self.measured.frequency)
        measured_features = self.measured_features
        
        return self.cost_fn(measured_features - model_features)
    
    @property
    def param_cost_function(self) -> Callable[[np.ndarray], float]:
        def cost_fn(theta):
            model = self.model.with_flat_params(theta)
            return self.cost(model)
        return cost_fn
    
    @property
    def model_cost_function(self) -> Callable[[Model], float]:
        # TODO update this to filter the parameters based on the 'fixed' flag
        def cost_fn(model):
            return self.cost(model)
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
        # callback_args = {
        #     'i_solver': 0,
        # }

        # def progress_callback(xk):            
        #     callback_args['i_solver'] += 1

        cost_fun = jax.jit(self.param_cost_function)

        # Run the minization routine
        result = scipy.optimize.minimize(cost_fun, x0, bounds=bounds, method='Nelder-Mead', options=options)

        self.model = self.model.with_params(result.x)
        return result


class OptaxFitter(FrequentistFitter):
    def run(self, 
            num_steps: int = 1000, 
            learning_rate: float = 1e-3, 
            *args, **kwargs):
        
        optim = optax.adam(learning_rate)
        model = self.model

        # Partition the model to get the initial tree of fittable parameters
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        loss = self.model_cost_function
        # loss = lambda model: self.cost(model)

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
    

class JaxNativeFitter(FrequentistFitter):
    def run(self, 
            *args, **kwargs):
        
        @eqx.filter_jit
        @eqx.filter_grad
        def cost_fn(theta):
            return self.param_cost_function(theta)
        
        # Populate bounds and options
        x0 = self.param_values_free
        
        options = {
            'maxiter': 10000
        }

        # Run the minization routine
        result = jax.scipy.optimize.minimize(cost_fn, x0, method='BFGS', options=options)

        self.model = self.model.with_params(result.x)
        return result