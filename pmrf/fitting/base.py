import glob
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import skrf
import jax.numpy as jnp

from pmrf.runner import BaseRunner
from pmrf.network_collection import NetworkCollection
from pmrf.models import Model
from pmrf.fitting.results import FitResults
from pmrf.frequency import Frequency
from pmrf import extract_features
from pmrf.io import save
from pmrf.util import RANK

class BaseFitter(BaseRunner, ABC):
    """
    Base class for all ParamRF fitters (frequentist and Bayesian).
    """
    def run(
        self,
        measured: str | skrf.Network | NetworkCollection,
        *,
        load_previous: bool = False, 
        new_uniform_frac: float | None = 0.1,
        save_model: bool = True,
        save_results: bool = True,
        plot: str | list[str] | None = 's_db',
        figure_subfolder: str | None = None,
        **kwargs        
    ) -> FitResults:
        """
        Executes the fit. Handles dynamic frequency extraction, delegates the mathematical 
        optimization, and packages the outputs into a FitResults object.
        """
        # 1. Try load from previous results
        if load_previous and self.output_path is not None:
            try:
                filename = glob.glob(f"{self.output_path}/*.hdf5")[0]
                results = FitResults.load_hdf(filename)
                self.logger.info("Loaded previous results.")
                return results
            except Exception:
                pass

        # 2. Prepare Measured Data & Frequency
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        measured = measured.copy()

        if self.frequency is None:
            if isinstance(measured, NetworkCollection):
                measured.interpolate_self()
                skrf_freq = measured.common_frequency()
            else:
                skrf_freq = measured.frequency
            self.frequency = Frequency.from_skrf(skrf_freq)

        # 3. Extract Target Features (NumPy arrays for the optimizer)
        # Note: JAX compilation is now entirely lazy and handled downstream
        target_features = extract_features(
            measured, self.frequency, self.features, sparam_kind=self.sparam_kind
        )
        
        # 4. Execute Optimization
        self.logger.info(f"Fitting {self.model.num_flat_params} parameters")
        self.logger.info(f"Parameter names: {self.model.flat_param_names()}")
        self.logger.info(f"Features: {self.features}")
        
        fitted_model, backend_results = self.optimize(target_features, **kwargs)

        # 5. Package Results
        results = FitResults()
        results.initial_model = self.model
        results.measured = measured
        results.fitted_model = fitted_model
        results.backend_results = backend_results
        results.backend_class = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        results.frequency = self.frequency
        results.features = self.features
        
        # Consolidate all kwargs for saving
        full_run_kwargs = kwargs.copy()
        full_run_kwargs.update({
            'load_previous': load_previous, 
            'new_uniform_frac': new_uniform_frac, 
            'save_model': save_model, 
            'save_results': save_results
        })
        results.run_kwargs = full_run_kwargs

        # 6. Post-processing logic
        if new_uniform_frac is not None:
            results.fitted_model = results.fitted_model.with_uniform_distributions(new_uniform_frac, respect_bounds=True)

        if plot is not None and not isinstance(plot, list):
            plot = [plot]

        save_output = self.output_path is not None and (save_model or save_results or plot is not None) and RANK == 0
        if save_output:
            output_prefix = f'{self.output_path}/{self.output_root}_' if self.output_root is not None else f'{self.output_path}/'
            
            if save_model:
                Path(self.output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info('Saving model...')
                save(Path(f'{output_prefix}fitted_model.prf').resolve(), results.fitted_model)

            if save_results:
                Path(self.output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info('Saving results...')
                results.save_hdf(Path(f'{output_prefix}results.hdf5').resolve())
        
            if plot is not None:
                self.logger.info('Plotting S-parameters...')
                figure_path = f'{self.output_path}/{figure_subfolder}' if figure_subfolder is not None else self.output_path
                figure_prefix = f'{figure_path}/{self.output_root}_' if self.output_root is not None else f'{figure_path}/'
                Path(figure_path).resolve().mkdir(parents=True, exist_ok=True)
                
                for plot_feature in plot:
                    func = getattr(results, f'plot_{plot_feature}')
                    func()
                    plt.savefig(Path(f'{figure_prefix}{plot_feature}.png').resolve(), dpi=400)
                    plt.close() # Close figure to prevent memory leak
        
        return results

    @abstractmethod
    def optimize(self, target_features: jnp.ndarray, **kwargs) -> tuple[Model, Any]:
        """Implemented by subclasses to perform the actual optimization algorithm."""
        pass