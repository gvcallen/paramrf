from abc import ABC, abstractmethod
from typing import Callable

import scipy
import scipy.optimize
import skrf

from pmrf.model import Model
from pmrf.parameter import Parameter, fixed
from pmrf.system import ModelSystem
from pmrf.fitting._results import BayesianResults, FrequentistResults

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
        param_infix = '_',
    ) -> None:
        if fit_frequency is not None:
            measured_interp = []
            for ntwk in measured:
                measured_interp.append(ntwk.interpolate(fit_frequency))
        else:
            raise ValueError("Error: Currently `fit_frequency` must be passed (i.e. all networks must be interpolated onto the same frequency) for fitting")
        
        self.model: Model | ModelSystem = model
        self.measured: skrf.Network | list[skrf.Network] = measured
        self.params: dict[str, Parameter] = params or {}
        self.param_infix = param_infix

    def _init_params(self):
        for name, value in self.model.params(separator=self.param_infix).items():
            # TODO add more complicated parameter initialization options e.g. default % width
            if not name in self.params:
                self.params[name] = fixed(value)

    @property
    def parameter_bounds(self) -> tuple[list, list]:
        pass
    
    @property
    def parameter_values(self) -> list[float]:
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass


class FrequentistFitter(BaseFitter):
    def make_cost_function(self) -> Callable:
        return self.model.make_cost_function(self.measured)


class ScipyFitter(FrequentistFitter):
    def fit(self, *args, **kwargs):
        cost_fn = self.make_cost_function()

        # Populate bounds and options
        x0 = self.parameter_values
        minimums, maximums = self.parameter_bounds
        bounds = scipy.optimize.Bounds(minimums, maximums)
        
        options = {
            'maxiter': 1000
        }

        # Setup the cost function lambda to pass to scipty
        callback_args = {
            'i_solver': 0,
        }

        def progress_callback(xk):
            callback_args['i_solver'] += 1
            print(callback_args['i_solver'])

        # Run the minization routine
        return scipy.optimize.minimize(cost_fn, args=callback_args, x0, bounds=bounds, method='SLSQP', options=options, callback=progress_callback)
