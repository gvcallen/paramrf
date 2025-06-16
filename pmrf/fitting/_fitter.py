from abc import ABC, abstractmethod
from typing import Callable

import scipy
import scipy.optimize
import skrf

import jax

from pmrf.model import Model
from pmrf.parameter import Parameter, fixed
from pmrf.system import ModelSystem
from pmrf.fitting._results import BayesianResults, FrequentistResults
from pmrf._numpy import numpy as np
from pmrf._features import FeatureExtractor

from pmrf._modifiers import ModifierChain

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
        features: list[FeatureExtractor] | FeatureExtractor | list[str] = None,
        param_infix = '_',
    ) -> None:
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

        self._init_params()

    def _init_params(self):
        model_params = self.model.params(separator=self.param_infix)
        for name, value in model_params.items():
            # TODO add more complicated parameter initialization options e.g. default % width
            if not name in self.params:
                self.params[name] = fixed(value)

    @abstractmethod
    def fit(self, *args, **kwargs):
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
    def make_cost_function(self) -> Callable:
        modifiers = ['L2', 'convolve-interleaved', 'L2', 'dB']
        cost_fn = self.model.make_cost_function(self.measured, features=self.features, modifiers=modifiers)
        def cost_fn_wrapper(x, *args, **kwargs):
            cost = cost_fn(x)
            # print(cost)
            return cost
        return cost_fn_wrapper
    
    def cost(self) -> float:
        return self.model.cost(self.measured)


class ScipyFitter(FrequentistFitter):
    def fit(self, *args, **kwargs):
        cost_fn = self.make_cost_function()

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
            # print(callback_args['i_solver'])

        # Run the minization routine
        cost_fn = jax.jit(cost_fn)

        # Free Parameters: ['coax_din', 'coax_dout', 'coax_len', 'coax_epr', 'coax_tand', 'coax_rho']

        # Fitting for 6 circuit model parameter(s)...
        # ********
        # [1.12e-03 3.20e-03 1.00e+01 1.45e+00 5.00e-02 1.60e-08]
        # ********
        # Bounds(array([8.96e-04, 2.56e-03, 8.00e+00, 1.16e+00, 0.00e+00, 1.28e-08]), array([1.344e-03, 3.840e-03, 1.200e+01, 1.740e+00, 1.000e-01, 1.920e-08]))
        # ********
        print('********')
        print(x0)
        print('********')
        print(bounds)
        print('********')

        # bounds = scipy.optimize.Bounds(np.array([9.0048e-04, 2.5728e-03, 8.0400e+00, 1.1658e+00, 1.0000e-03, 1.2864e-08]), np.array([1.33952e-03, 3.82720e-03, 1.19600e+01, 1.73420e+00, 9.90000e-02, 1.91360e-08]))
        bounds = scipy.optimize.Bounds(np.array([8.96e-04, 2.56e-03, 8.00e+00, 1.16e+00, 0.00e+00, 1.28e-08]), np.array([1.344e-03, 3.840e-03, 1.200e+01, 1.740e+00, 1.000e-01, 1.920e-08]))

        # ********
        # [1.12e-03 3.20e-03 1.00e+01 1.45e+00 5.00e-02 1.60e-08 1.00e+00]
        # ********
        # Bounds(array([9.0048e-04, 2.5728e-03, 8.0400e+00, 1.1658e+00, 1.0000e-03,
        #        1.2864e-08, 1.0000e+00]), array([1.33952e-03, 3.82720e-03, 1.19600e+01, 1.73420e+00, 9.90000e-02,
        #        1.91360e-08, 1.00000e+00]))
        # ********        

        result = scipy.optimize.minimize(cost_fn, x0, args=callback_args, bounds=bounds, method='Nelder-Mead', options=options, callback=progress_callback)

        # print(result)
        self.model = self.model.with_params(result.x)

        return result
