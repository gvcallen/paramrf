from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
import glob

import numpy as np
import jax.numpy as jnp

from pmrf.runner import BaseRunner
from pmrf.models.model import Model
from pmrf.util import LivePlotter, RANK
from pmrf.sampling.results import SampleResults 

class BaseSampler(BaseRunner, ABC):
    """
    Base class for model sampling and active learning loops.
    """
    def __init__(self, model: Model, **kwargs):
        if model.num_flat_params == 0:
            raise ValueError("Model has no free parameters to sample.")
            
        super().__init__(model, **kwargs)
        
        self.sampled_params: jnp.ndarray | None = None
        self.sampled_features: jnp.ndarray | None = None
        self.feature_plotters: list[LivePlotter] = []

    def run(
        self, 
        *, 
        load_previous: bool = False, 
        save_results: bool = True, 
        **kwargs
    ) -> SampleResults:
        """
        Executes the sampling process. Handles state delegation and packages 
        the outputs into a SampleResults object.
        """
        # 1. Try load from previous results
        if load_previous and self.output_path is not None:
            try:
                filename = glob.glob(f"{self.output_path}/*.hdf5")[0]
                results = SampleResults.load_hdf(filename)
                self.logger.info("Loaded previous sampling results.")
                return results
            except Exception:
                pass

        # 2. Execute Sampling (Delegates to Strategy/Backend)
        self.logger.info(f"Sampling {self.model.num_flat_params} parameters...")
        self.logger.info(f"Parameter names: {self.model.flat_param_names()}")
        
        samples, backend_results = self.sample(**kwargs)

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
        if self.output_path is not None and save_results and RANK == 0:
            Path(self.output_path).resolve().mkdir(parents=True, exist_ok=True)
            output_prefix = f'{self.output_path}/{self.output_root}_' if self.output_root is not None else f'{self.output_path}/'
            self.logger.info('Saving sampling results...')
            results.save_hdf(Path(f'{output_prefix}results.hdf5').resolve())
            
        return results

    @abstractmethod
    def sample(self, **kwargs) -> tuple[jnp.ndarray, Any]:
        """
        Implemented by subclasses to perform the actual sampling algorithm.
        Returns a tuple of (thetas, backend_results).
        """
        pass

    def add_samples(self, theta: jnp.ndarray, plot: list[str] | None = None) -> jnp.ndarray | None:
        """Adds samples, computes features via lazy compilation, and plots."""
        if plot is not None and isinstance(plot, str):
            plot = [plot]

        new_thetas = jnp.atleast_2d(theta)
        N, D = new_thetas.shape
        new_features = None
        
        # Ensure we only try to extract features if frequency/features are defined
        if N > 0 and self.frequency is not None and self.features is not None:
            time_str = datetime.now().strftime("%H:%M:%S")
            num_existing = len(self.sampled_params) if self.sampled_params is not None else 0
            
            if N == 1:
                self.logger.info(f"Computing sample #{num_existing + 1} at {time_str}")
            else:
                self.logger.info(f"Computing samples #{num_existing + 1}-{num_existing + N} at {time_str}")
            
            # Use the lazy accessor from BaseRunner
            new_features = self.extract_model_features(new_thetas)

        # Update State
        if self.sampled_params is not None:
            self.sampled_params = jnp.vstack((self.sampled_params, new_thetas))
            if new_features is not None:
                self.sampled_features = jnp.vstack((self.sampled_features, new_features))
        else:
            self.sampled_params = new_thetas
            self.sampled_features = new_features

        # I/O Checkpoint (numpy fallback for crash recovery during active learning)
        if self.output_path is not None and RANK == 0:
            np.save(f"{self.output_path}/params.npy", np.asarray(self.sampled_params))
            if self.sampled_features is not None:
                np.save(f"{self.output_path}/features.npy", np.asarray(self.sampled_features))

        # Live Plotting
        if plot is not None and new_features is not None:
            for i, feature_name in enumerate(plot):
                if i >= len(self.feature_plotters):
                    self.feature_plotters.append(
                        LivePlotter(title=f"{feature_name}", xlabel=f"Frequency ({self.frequency.unit})", ylabel=f"{feature_name}")
                    )
                
                plotter = self.feature_plotters[i]
                for current_theta, current_feature in zip(new_thetas, new_features):
                    param_dict = {k: float(v) for k, v in zip(self.model.flat_param_names(), current_theta)}
                    plotter.ax.set_title(f"{feature_name} (num_samples = {len(self.sampled_params)})")
                    plotter.add_curve(str(param_dict), y_values=np.real(current_feature), x_values=self.frequency.f_scaled)
                    
        return new_features