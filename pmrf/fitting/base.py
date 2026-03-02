import os
from typing import Sequence
import glob
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import skrf
import jax.numpy as jnp

from pmrf.runner import BaseRunner
from pmrf.constants import FeatureSpecT
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
        output_path: str | None = None,
        output_root: str | None = None,
        plot: FeatureSpecT | None = None,
        save_model: bool = True,
        save_results: bool = True,
        figure_dir: str | None = None,
        fitted_uniform_frac: float | None = 0.1,
        **kwargs        
    ) -> tuple[Model, FitResults]:
        """
        Executes the fit. Handles dynamic frequency extraction, delegates the mathematical 
        optimization, and packages the outputs into a FitResults object.
        """
        self.output_path = output_path
        self.output_root = output_root
        self.plot_features = plot
        
        if output_path is not None and RANK == 0:
            os.makedirs(output_path, exist_ok=True)        
        
        # Prepare Measured Data & Frequency
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

        # Lazily update features. Delete the feature function (for now TODO clean this up)
        self._feature_fn = None
        features = self.features
        if not isinstance(features, Sequence) and not isinstance(features, dict):
            features = [features]
        if isinstance(measured, NetworkCollection) and not isinstance(features, dict):
            features = {ntwk.name: features for ntwk in measured}
        if isinstance(measured, NetworkCollection) and not isinstance(features, dict):
            features = {ntwk.name: features for ntwk in measured}            
        self.features = features
        
        # Extract Target Features (NumPy arrays for the optimizer)
        # Note: JAX compilation is now entirely lazy and handled downstream
        target_features = extract_features(
            measured, self.frequency, self.features, **self.feature_kwargs,
        )
        
        # Execute Optimization
        self.logger.info(f"Fitting {self.model.num_flat_params} parameters")
        self.logger.info(f"Parameter names: {self.model.flat_param_names()}")
        self.logger.info(f"Features: {self.features}")
        
        fitted_model, backend_results = self.optimize(target_features, **kwargs)

        # Package Results
        results = FitResults()
        results.initial_model = self.model
        results.measured = measured
        results.fitted_model = fitted_model
        results.backend_results = backend_results
        results.backend_class = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        results.frequency = self.frequency
        results.features = self.features
        results.run_kwargs = kwargs

        # 6. Post-processing logic
        if fitted_uniform_frac is not None:
            results.fitted_model = results.fitted_model.with_uniform_distributions(fitted_uniform_frac, respect_bounds=True)

        if plot is not None and not isinstance(plot, list):
            plot = [plot]

        save_output = output_path is not None and (save_model or save_results or plot is not None) and RANK == 0
        if save_output:
            output_prefix = f'{output_path}/{output_root}_' if output_root is not None else f'{output_path}/'
            
            if save_model:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info('Saving model...')
                save(Path(f'{output_prefix}fitted_model.prf').resolve(), results.fitted_model)

            if save_results:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info('Saving results...')
                results.save_hdf(Path(f'{output_prefix}fit_results.hdf5').resolve())
        
            if plot is not None:
                self.logger.info('Plotting results...')
                figure_path = f'{output_path}/{figure_dir}' if figure_dir is not None else output_path
                figure_prefix = f'{figure_path}/{output_root}_' if output_root is not None else f'{figure_path}/'
                Path(figure_path).resolve().mkdir(parents=True, exist_ok=True)
                
                for plot_feature in plot:
                    func = getattr(results, f'plot_{plot_feature}')
                    func()
                    plt.savefig(Path(f'{figure_prefix}{plot_feature}.png').resolve(), dpi=400)
                    plt.close()
        
        return results.fitted_model, results

    @abstractmethod
    def optimize(
        self,
        target: jnp.ndarray,
        **kwargs
    ) -> tuple[Model, Any]:
        """Implemented by subclasses to perform the actual optimization algorithm."""
        pass