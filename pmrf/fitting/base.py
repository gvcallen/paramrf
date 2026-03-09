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
from pmrf.models.model import Model
from pmrf.fitting.results import FitResults
from pmrf.frequency import Frequency
from pmrf import extract_features
from pmrf.io import save
from pmrf.util import RANK

class BaseFitter(BaseRunner, ABC):
    r"""
    Base class for all ParamRF fitters (frequentist and Bayesian).
    
    This runner fits model parameters to measured RF data (like S-parameters 
    or Y-parameters). It handles extracting the features you want to fit, 
    runs the specific optimization algorithm, and returns the final model.

    .. rubric:: Main methods

    .. autosummary::
       :nosignatures:
       
       run
       execute

    Parameters
    ----------
    model : Model
        The initial ParamRF model containing the free parameters to be optimized.
    frequency : Frequency, optional
        The frequency band to fit over. If not provided, it is automatically 
        extracted from the measured data during the run.
    features : FeatureSpecT, optional
        The target features to extract from both the model and the measured data 
        for the objective function.
    **feature_kwargs
        Additional keyword arguments passed directly to 
        :meth:`pmrf.extract_features`.
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
        fitted_uniform_frac: float | None = 0.0,
        **kwargs        
    ) -> tuple[Model, FitResults]:
        r"""
        Run the fitting process against the provided measurement data.
        
        This method automatically sets up the frequency, extracts the target features, 
        and calls the subclass's ``execute`` method. It also handles saving the 
        results and generating plots if requested.
        
        **Note:** Any extra keyword arguments (**kwargs) are passed directly to the 
        underlying ``execute`` method.

        Parameters
        ----------
        measured : str or skrf.Network or NetworkCollection
            The ground-truth measurement data to fit the model against. Can be a 
            path to a Touchstone file, a scikit-rf Network, or a collection.
        output_path : str, optional
            The directory where results, models, and figures should be saved.
        output_root : str, optional
            A prefix string appended to all saved filenames.
        plot : FeatureSpecT, optional
            Specific features to plot and save as PNG images after fitting.
        save_model : bool, default=True
            Whether to save the fitted model to disk.
        save_results : bool, default=True
            Whether to save the full ``FitResults`` object to an HDF5 file.
        figure_dir : str, optional
            A sub-directory within ``output_path`` specifically for saved figures.
        fitted_uniform_frac : float, optional, default=0.1
            If provided, the final fitted model's parameters will be assigned 
            uniform distributions spanning $\pm$ this fraction around the optimal 
            values (e.g., $0.1$ implies $\pm 10\%$). Set to ``None`` to skip.
        **kwargs
            Additional arguments passed directly to the subclass's ``execute`` method.

        Returns
        -------
        tuple[Model, FitResults]
            A tuple containing the fitted ParamRF model and the results object.
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
        self.logger.debug(f"Parameter names: {self.model.flat_param_names()}")
        self.logger.debug(f"Features: {self.features}")
        
        fitted_model, backend_results = self.execute(target_features, **kwargs)

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
    def execute(
        self,
        target: jnp.ndarray,
        **kwargs
    ) -> tuple[Model, Any]:
        r"""
        Implemented by subclasses to run the specific optimization algorithm.
        
        Parameters
        ----------
        target : jax.numpy.ndarray
            The extracted target features to fit against.
        **kwargs
            Backend-specific algorithm parameters passed down from ``run()``.
            
        Returns
        -------
        tuple[Model, Any]
            The fitted model and the raw results from the solver.
        """
        pass