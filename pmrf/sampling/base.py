from abc import ABC, abstractmethod
import logging
import os
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.constants import FeatureInputT
from pmrf.features import extract_features
from pmrf.util import LivePlotter
from pmrf.sampling.results import SampleResults, SampleSettings

class BaseSampler(ABC):
    def __init__(
        self,
        model: Model,
        *,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        output_path: str | None = None,
        batch_size: int = 1,
    ):
        if model.num_flat_params == 0:
            raise Exception("Model has no free parameters to sample")
        
        if features is not None and frequency is None:
            raise Exception("Cannot sample features without a frequency")

        self.model: Model = model
        self.frequency: Frequency = frequency
        self.features: FeatureInputT = features
        self.logger = logging.getLogger(__name__)
        self.output_path: str = output_path
        self.batch_size: int = batch_size
        
        self.sampled_params: jnp.ndarray = None
        self.sampled_features: jnp.ndarray = None
        self.feature_plotters: list[LivePlotter] = None    
        
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)

    def add_samples(self, theta: jnp.ndarray, plot=None) -> jnp.ndarray | None:
        """
        Adds samples to the sampler.
        
        This computes any requested sample features, and also updates the feature plots.
        Handles both single samples (D,) and simultaneous batches (N, D).

        Parameters
        ----------
        theta : jnp.ndarray
            The sample parameters to add. 
            If shape (D,), treats as a single sample.
            If shape (N, D), treats as a batch of N samples to process simultaneously.
            
        plot : list[str] | None
            Features to plot. Need not be the same as features specified upon sampler creation.
            
        Returns
        -------
        jnp.ndarray | None
            The features for the added samples. Returns shape (F,) for single input
            or (N, F) for batch input.
        """
        if plot is not None and isinstance(plot, str):
            plot = [plot]

        # Normalize input shape
        new_thetas = jnp.atleast_2d(theta)
        N, D = new_thetas.shape
        new_features = None
        
        # Compute features
        if N > 0 and self.features is not None:
            time_str = datetime.now().strftime("%H:%M:%S")
            num_existing_samples = len(self.sampled_params) if self.sampled_params is not None else 0
            if N == 1:
                self.logger.info(f"Computing sample #{num_existing_samples + 1} at {time_str}")
            else:
                self.logger.info(f"Computing samples #{num_existing_samples + 1}-{num_existing_samples + N} at {time_str}")
            
            if self.logger.level >= logging.DEBUG:
                for theta in new_thetas:
                    printable_params = {k: round(float(v), 2) for k, v in zip(self.model.flat_param_names(), theta)}
                    self.logger.debug(f"Parameters = {printable_params}")
            new_features = self.feature_fn(new_thetas)

        # Update samples and features
        if self.sampled_params is not None:
            self.sampled_params = jnp.vstack((self.sampled_params, new_thetas))
            self.sampled_features = jnp.vstack((self.sampled_features, new_features))
        else:
            self.sampled_params = new_thetas
            self.sampled_features = new_features

        # Save to disk
        if self.output_path is not None:
            np.save(f"{self.output_path}/params.npy", self.sampled_params)
            np.save(f"{self.output_path}/features.npy", self.sampled_features)

        # Plot
        if plot is not None:
            if self.feature_plotters is None:
                self.feature_plotters = []
            
            for i, feature_name in enumerate(plot):
                if i >= len(self.feature_plotters):
                    self.feature_plotters.append(LivePlotter(title=f"{feature_name}", xlabel=f"Frequency ({self.frequency.unit})", ylabel=f"{feature_name}"))
                plotter = self.feature_plotters[i]
            
                new_plot_features = self.feature_fn(new_thetas, features=feature_name)
                for current_theta, current_feature in zip(new_thetas, new_plot_features):
                    param_dict = dict(zip(self.model.flat_param_names(), current_theta.tolist()))
                    plotter.ax.set_title(f"{feature_name} (num_samples = {len(self.sampled_params)})")
                    plotter.add_curve(f"{param_dict}", y_values=jnp.real(current_feature), x_values=self.frequency.f_scaled)

    def inverse_cumulative_distribution_fn(self, u, as_numpy=False) -> jnp.ndarray:
        """
        Create the inverse cumulative distribution function (unit hypercube to parameter space).

        Parameters
        ----------
        as_numpy : bool, optional, default=False
            If True, returns a function compatible with NumPy arrays; otherwise JAX arrays.

        Returns
        -------
        callable
            Function transforming a unit hypercube vector `u` to parameter vector `theta`.
        """
        model_distribution = self.model.distribution()
        
        @jax.jit
        def icdf_fn(u):
            return model_distribution.icdf(u)
            
        if as_numpy:
            icdf_transform_fn_jax = icdf_fn
            icdf_fn = lambda hypercube: np.array(icdf_transform_fn_jax(hypercube))
        
        return icdf_fn(u)
    
    def cumulative_distribution_fn(self, theta, as_numpy=False) -> jnp.ndarray:
        """
        Create the cumulative distribution function (parameter space to unit hypercube).

        Parameters
        ----------
        as_numpy : bool, optional, default=False
            If True, returns a function compatible with NumPy arrays; otherwise JAX arrays.

        Returns
        -------
        callable
            Function transforming a parameter vector `theta` to a unit hypercube vector `u`.
        """
        model_distribution = self.model.distribution()
        
        @jax.jit
        def cdf_fn(theta):
            return model_distribution.cdf(theta)
            
        if as_numpy:
            cdf_fn_jax = cdf_fn
            cdf_fn = lambda hypercube: np.array(cdf_fn_jax(hypercube))
        
        return cdf_fn(theta)
        
    def feature_fn(self, thetas: jnp.ndarray, *, model=None, features=None) -> jnp.ndarray:
        """
        Create a JIT-compiled function to extract features from model parameters.
        
        The features are used for feature generation and adaptive stopping criteria/sample proposal.

        Parameters
        ----------
        as_numpy : bool, default=False
            If True, the returned function handles NumPy arrays; otherwise JAX arrays.

        Returns
        -------
        callable
            A function taking ``theta`` and returning feature values.
        """
        features = features if features is not None else self.features
        model = model if model is not None else self.model
        
        if self.frequency is None or features is None:
            raise Exception('Cannot make a feature function without a sampling frequency or features')
        
        def general_feature_fn(theta):
            model = self.model.with_params(theta)
            return extract_features(model, self.frequency, features)
            
        if thetas.ndim > 1:
            feature_fn_final = jax.jit(jax.vmap(general_feature_fn))
        else:
            feature_fn_final = jax.jit(general_feature_fn)
        return feature_fn_final(thetas)
    
    @abstractmethod
    def run(self, *args, **kwargs) -> tuple[list[Model], SampleResults]:
        """Entry point for generating models."""
        pass