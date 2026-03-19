from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
import glob

import numpy as np
import jax.numpy as jnp

from pmrf.runner import BaseRunner
from pmrf.core import Model
from pmrf.utils import LivePlotter, RANK
from pmrf.sample.results import SampleResults 
from pmrf.features import extract_features

class BaseSampler(BaseRunner, ABC):
    r"""
    Base class for model sampling and active learning loops.
    
    This runner is designed to explore the parameter space of a model, either 
    through pre-defined experimental designs or adaptive active learning strategies. 
    It manages state caching, crash-recovery checkpoints, and live plotting of 
    extracted features during the sampling loop.

    .. rubric:: Methods

    .. autosummary::
       :nosignatures:

       run
       execute
       update

    Parameters
    ----------
    model : :class:`~pmrf.model.Model`
        The base ParamRF model to be sampled.
    **kwargs
        Additional arguments passed up to :class:`~pmrf.runner.BaseRunner` 
        (e.g., ``frequency``, ``features``).
    """
    expensive: bool = False
    
    def __init__(self, model: Model, **kwargs):            
        super().__init__(model, **kwargs)
        
        self.sampled_params: jnp.ndarray | None = None
        self.sampled_features: jnp.ndarray | None = None
        
        # Context variables
        self.plot_features = None
        self.feature_plotters: list[LivePlotter] = []

    def run(
        self, 
        *,
        plot = None,
        output_path: str | None = None,
        load_previous: bool = False, 
        save_results: bool = True, 
        save_figures: bool = True,
        **kwargs
    ) -> SampleResults:
        r"""
        Execute the sampling process.
        
        This method handles plotting setup, optional recovery from previous runs, 
        delegates the core algorithm to the subclass's :meth:`sample` method, 
        and packages the final results.

        Parameters
        ----------
        plot : str or list[str], optional
            Specific features to plot live during the sampling loop and save 
            as PNG images.
        output_path : str, optional
            The directory where results, numpy crash-checkpoints, and figures 
            should be saved.
        load_previous : bool, default=False
            If ``True`` and ``output_path`` is provided, it will attempt to 
            load and return an existing ``sample_results.hdf5`` file instead 
            of re-running the sampling.
        save_results : bool, default=True
            Whether to save the final ``SampleResults`` object to an HDF5 file.
        save_figures : bool, default=True
            Whether to save the generated plots to disk.
        **kwargs
            Additional arguments passed directly to the subclass's :meth:`sample` method.

        Returns
        -------
        :class:`~pmrf.sample.results.SampleResults`
            The comprehensive results object containing the evaluated parameters 
            and their corresponding extracted features.
        """
        # Updates feature variables
        if plot is not None and isinstance(plot, str):
            plot = [plot]        
        
        self.output_path = output_path
        self.plot_features = plot
        
        if self.plot_features is not None:
            for i, feature_name in enumerate(plot):
                if i >= len(self.feature_plotters):
                    self.feature_plotters.append(
                        LivePlotter(title=f"{feature_name}", xlabel=f"Frequency ({self.frequency.unit})", ylabel=f"{feature_name}")
                    )                
        
        # 1. Try load from previous results
        if load_previous and output_path is not None:
            try:
                filename = glob.glob(f"{output_path}/*.hdf5")[0]
                results = SampleResults.load_hdf(filename)
                self.logger.info("Loaded previous sampling results.")
                return results
            except Exception:
                pass

        # 2. Execute Sampling (Delegates to Strategy/Backend)
        self.logger.info(f"Sampling {self.model.num_flat_params} parameters...")
        self.logger.debug(f"Parameter names: {self.model.flat_param_names()}")
        self.logger.debug(f"Features: {self.features}")
        
        samples, backend_results = self.execute(output_path=output_path, **kwargs)

        # 3. Package Results
        results = SampleResults()
        results.initial_model = self.model
        results.frequency = self.frequency
        results.features = self.features
        
        # We store the accumulated state from add_samples or the final return
        results.sampled_params = self.sampled_params if self.sampled_params is not None else samples
        results.sampled_features = self.sampled_features
        
        results.backend_results = backend_results
        results.backend_class = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        
        # Consolidate kwargs
        full_run_kwargs = kwargs.copy()
        full_run_kwargs.update({'load_previous': load_previous, 'save_results': save_results})
        results.run_kwargs = full_run_kwargs

        # 4. Save Output
        if output_path is not None and RANK == 0:
            if save_results or save_figures:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
            
            if save_results:            
                self.logger.info('Saving results...')
                results.save_hdf(Path(f'{output_path}/sample_results.hdf5').resolve())
                
            if save_figures and self.plot_features is not None:
                self.logger.info('Saving figures...')
                for feature, plotter in zip(self.plot_features, self.feature_plotters):
                    plotter.fig.savefig(f"{output_path}/{feature}.png", dpi=400)
            
        return results    

    def update(self, theta: jnp.ndarray) -> None:
        r"""
        Evaluate new parameters and update the internal sampling state.
        
        This function is central to active learning and iterative sampling backends. 
        It extracts the features for the provided parameters, appends the data to 
        the internal tracking arrays, updates any configured live plots, and 
        (if ``expensive=True``) automatically saves intermediate crash-recovery 
        checkpoints to disk using native NumPy serialization.

        Parameters
        ----------
        theta : jax.numpy.ndarray
            A 1D array of a single parameter set, or a 2D array representing 
            a batch of parameters to simulate.
        """
        new_thetas = jnp.atleast_2d(theta)
        N, D = new_thetas.shape
        new_features = None
        
        # Ensure we only try to extract features if frequency/features are defined
        if N > 0 and self.frequency is not None and self.features is not None:
            time_str = datetime.now().strftime("%H:%M:%S")
            num_existing = len(self.sampled_params) if self.sampled_params is not None else 0
            
            if self.expensive:
                if N == 1:
                    self.logger.info(f"Simulating sample #{num_existing + 1} at {time_str}")
                else:
                    self.logger.info(f"Simulating samples #{num_existing + 1}-{num_existing + N} at {time_str}")
            new_features = super().model_features(new_thetas)

        # Update State
        if self.sampled_params is not None:
            self.sampled_params = jnp.vstack((self.sampled_params, new_thetas))
            if new_features is not None:
                self.sampled_features = jnp.vstack((self.sampled_features, new_features))
        else:
            self.sampled_params = new_thetas
            self.sampled_features = new_features

        # I/O Checkpoint (numpy fallback for crash recovery during active learning)
        if self.expensive and self.output_path is not None and RANK == 0:
            np.save(f"{self.output_path}/sampled_params.npy", np.asarray(self.sampled_params))
            if self.sampled_features is not None:
                np.save(f"{self.output_path}/sampled_features.npy", np.asarray(self.sampled_features))
                
        # Live Plotting
        if self.plot_features is not None:
            for plotter, plot_feature in zip(self.feature_plotters, self.plot_features):
                for current_theta in new_thetas:
                    m = self.model.with_params(current_theta)
                    plot_features = extract_features(m, self.frequency, plot_feature)
                    
                    param_dict = {k: float(v) for k, v in zip(self.model.flat_param_names(), current_theta)}
                    plotter.ax.set_title(f"{plot_feature}")
                    plotter.add_curve(str(param_dict), y_values=np.real(plot_features), x_values=self.frequency.f_scaled)
                    
    @abstractmethod
    def execute(
        self,
        **kwargs,
    ) -> tuple[jnp.ndarray, Any]:
        r"""
        Implemented by subclasses to perform the actual sampling algorithm.

        Parameters
        ----------
        **kwargs
            Backend-specific algorithm parameters passed down from :meth:`run`.

        Returns
        -------
        tuple[jax.numpy.ndarray, Any]
            A tuple containing the generated array of parameters (thetas) and 
            any raw backend-specific results or metadata object.
        """
        pass