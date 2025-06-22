from functools import partial
from abc import ABC, abstractmethod
from typing import Callable

import skrf
import scipy
import jax
import equinox as eqx
import optax

import pmrf.numpy as np
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf.parameters import Parameter, fixed
from pmrf.numpy import USE_JAX
from pmrf.functions import mag_2_db, convolve_interleaved

class BaseFitter(ABC):
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        params: dict[str, Parameter] | None = None,
        frequency: skrf.Frequency | None = None,
        features: list[str] | list[tuple[str, tuple]] = ['s'],
        param_infix = '_'
    ) -> None:
        """The base fitter initializer.

        Args:
            model (Model):                                              The model to fit.
            measured (skrf.Network | list[skrf.Network]):               The measured networks to fit against. If a list is passed, the networks are viewed as being part of a large, stacked N-port network.
                                                                        If a measurement is not available, an empty network can be passed. See `SystemModel` for an example use-case.
            params (dict[str, Parameter] | None, optional):             Parameters for the models, specified in a flattened format. See `param_infix`. Defaults to `None`, in which case all parameters are set as normal with 5% standard deviation. (Not yet implemented)
            frequency (skrf.Frequency | None, optional):                The frequency to fit against. Defaults to `None`.
            features (list[str] | list[tuple[str, tuple]], optional):   The features to extract from the models and networks for cost functions, likelihoods etc.
                                                                        Each string is a function or property of the model or network respectively (e.g. 's_db', 's_mag' etc.), and `ports` are the ports to use as a tuple (e.g. (0, 0)).
                                                                        If a list of strings is passed, the features are extracted for each port within in each network/model and stacked column-wise into a "feature matrix".
                                                                        If a list of strings-tuple pairs are passed, then each feature is extracted for each port individually, where port numbers are for the full model (e.g. the stacked network in the case where a list of measurements are passed).
            param_infix (str, optional):                                The infix between submodels for flattened parameters. Parameters are specified as for `{model}{infix}{submodel1}{infix}{submodel2}{...}{infix}{param}`. Defaults to '_'.
        """
        # Currently, all frequencies must be the same across all measurements
        measured = [measured] if not isinstance(measured, list) else measured
        if frequency is not None:
            measured = [ntwk.interpolate(frequency) for ntwk in measured]
        else:
            freq = measured[0].frequency
            for ntwk in measured:
                if ntwk.frequency != freq and not len(ntwk.frequency) == 0:
                    raise ValueError("Error: Currently `fit_frequency` must be passed for multi-measurement fits (i.e. all networks must be explicitly interpolated onto the same frequency for fitting)")
                
        # Make features the correct format
        features_new = []
        if isinstance(features[0], str):
            p = 0
            for ntwk in measured:
                for ports in ntwk.port_tuples:
                    for feature in features:
                        features_new.append((feature, ports))
                    p += 1
        features = features_new
        
        # The most important part - making sure our model params are in the same order as the model's
        user_params = params
        model_params = self.model.params(separator=param_infix)
        final_params = {}
        for name, value in model_params.items():
            # TODO add more complicated parameter initialization options e.g. default % width.
            # Also add error check if the user passes a parameter that is NOT in the model
            final_params[name] = user_params[name] if name in user_params else fixed(value)

        self.model: Model = model
        self.measured: list[skrf.Network] = measured
        self.params: dict[str, Parameter] = params or {}
        self.param_infix = param_infix
        self.features = features
        self.measured_features = None # we extract features before each fit
        self.params = final_params
        
    def extract_features(self, source: Model | skrf.Network | list[skrf.Network], freq: Frequency = None) -> np.ndarray:
        if freq is None:
            freq = source.frequency
        features = self.features

        n_frequencies = len(freq)
        n_features = len(features)

        X = np.zeros((n_frequencies, n_features), dtype=np.complex128)
        for d, feature in enumerate(features):
            prop = feature[0]
            m, n = feature[1]
            x = None
            
            if isinstance(source, Model):
                x = getattr(source, prop)(freq)[:,m,n]
            elif isinstance(source, skrf.Network):
                x = getattr(source, prop)[:,m,n]
            elif isinstance(source, list):
                p = 0
                for ntwk in source:
                    nports = ntwk.nports
                    if m >= p + nports:
                        p += nports
                        continue
                    x = ntwk.s[:, m-p, n-p]
                if x is None:
                    raise Exception('Error: port of out bounds')
                
            if USE_JAX:
                X = X.at[:, d].set(x)
            else:
                X[:, d] = x
        return X        

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @property
    def params(self) -> list[str]:
        return dict(self.params)
    
    @property
    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([v.lower for v in self.params.values()])
        upper = np.array([v.upper for v in self.params.values()])
        return lower, upper
    
    @property
    def free_param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([v.lower for v in self.params.values() if not v.fixed])
        upper = np.array([v.upper for v in self.params.values() if not v.fixed])
        return lower, upper    
    
class FrequentistFitter(BaseFitter):
    def __init__(self, cost: list[Callable[[np.ndarray], np.ndarray]] | eqx.Module = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if cost is None:
            L2 = partial(np.linalg.norm, order=2)
            cost = [L2, partial(convolve_interleaved, axis=1), L2, mag_2_db]

        self.cost_fn = eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in cost])
        self.measured_features = self.extract_features(self.measured)

    def cost(self, model: Model | None = None) -> np.ndarray:
        model = model or self.model
        features = self.extract_features(model, self.measured.frequency)
        return self.cost_fn(self.measured_features - features)
    
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
    def run(self, *args, **kwargs):
        # Populate bounds and options
        x0 = self.param_values_free
        minimums, maximums = self.free_param_bounds
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