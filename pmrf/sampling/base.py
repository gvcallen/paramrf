from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from typing import Any
import os
from datetime import datetime

import numpy as np
import skrf
import jax
from jax import flatten_util
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.models.model import Model, wrap
from pmrf.constants import FeatureInputT, FeatureT
from pmrf._features import extract_features
from pmrf._util import LivePlotter

@dataclass
class SampleSettings:
    """
    Configuration settings for the sampling process.

    Attributes
    ----------
    frequency : Frequency or None
        The frequency grid used for the feature extraction.
    features : list of FeatureT or None
        The list of features extracted for during sampling.
    """
    frequency: Frequency | None = None
    features: list[FeatureT] | None = None

@dataclass
class SampleResults:
    """
    Container for the results of a model sampling process.
    
    Attributes
    ----------
    initial_model : Model or None
        The model with the initial parameters.
    sampled_models : list[Model] or None
        The sampled models.
    algorithm_results : Any
        The raw result object returned by the sampling backend/algorithm.
    settings : FitSettings or None
        The configuration used to execute the fit.
    """    
    initial_model: Model | None = None
    sampled_models: list[Model] | None = None
    sampled_params: jnp.ndarray | None = None
    sampled_features: jnp.ndarray | None = None
    backened_results: Any = None
    settings: SampleSettings | None = None

class BaseSampler(ABC):
    def __init__(
        self,
        model: Model,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        output_path: str | None = None,
        jit_features: bool = False, 
    ):
        if model.num_flat_params == 0:
            raise Exception("Model has no free parameters to sample")
        
        self.model: Model = model
        self.frequency: Frequency = frequency
        self.features: FeatureInputT = features
        self.logger = logging.getLogger(__name__)
        self.output_path: str = output_path
        self.jit_features: bool = jit_features
        
        self.sampled_params: jnp.ndarray = None
        self.sampled_features: jnp.ndarray = None
        self.feature_plotters: list[LivePlotter] = None
        
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            
    def add_sample(self, theta: jnp.ndarray, plot=None) -> jnp.ndarray | None:
        """
        Adds a sample to the sampler.
        
        This computes any requested sample features, and also updates the feature plot.

        Parameters
        ----------
        theta : jnp.ndarray
            The sample parameters to add.
        plot : list[str] | None
            Features to plot. Need not be the same as features specified upon sampler creation.
            
        Returns
        ----------
        jnp.ndarray | None
            The features for this sample, if any.
        """        
        # Check if we already have this sample
        existing_idx = None
        if self.sampled_params is not None:
            for i, sampled_theta in enumerate(self.sampled_params):
                if jnp.all(jnp.equal(sampled_theta, theta)):
                    existing_idx = i
                    break
        
        # Compute the sample and features, if necessary, otherwise just retrieve them
        new_params_dict = dict(zip(self.model.flat_param_names(), theta.tolist()))
        new_features = None
        if existing_idx is None:
            sample_count = len(self.sampled_params) if self.sampled_params is not None else 0
        
            # Compute the sample features
            time_str = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"Computing sample #{sample_count + 1} at {time_str} with params "
                f"{{ {', '.join([f'{k}: {v:.2f}' for k, v in new_params_dict.items()])} }} "
            )
            if self.features is not None:
                new_features = self.feature_fn(theta)
        
            # Add the params and features to self and save
            if self.sampled_params is None:
                self.sampled_params = theta[None, ...]
                if self.features is not None:
                    self.sampled_features = new_features[None, ...]
            else:
                self.sampled_params = jnp.vstack((self.sampled_params, theta[None, ...]))
                if self.features is not None:
                    self.sampled_features = jnp.vstack((self.sampled_features, new_features[None, ...]))
                
            if self.output_path is not None:
                np.save(f"{self.output_path}/params.npy", self.sampled_params)
                np.save(f"{self.output_path}/features.npy", self.sampled_features)
        else:
            if self.features is not None:
                new_features = self.sampled_features[existing_idx]
        
        # Deal with plotting
        if plot is not None:
            # Create plotters (lazily)
            if self.feature_plotters is None:
                self.feature_plotters = []
                for p in plot:
                    self.feature_plotters.append(LivePlotter(title=f"{p}", xlabel=f"Frequency ({self.frequency.unit})", ylabel=f"{p}"))
            
            # Extract the plot features
            new_plot_features = self.feature_fn(theta, features=plot)
            
            for i, (plotter, feature_name) in enumerate(zip(self.feature_plotters, plot)):
                feature = new_plot_features[..., i]
                plotter.ax.set_title(f"{feature_name} (num_samples = {len(self.sampled_params)})")
                plotter.add_curve(f"{new_params_dict}", y_values=jnp.real(feature), x_values=self.frequency.f_scaled)
                
        return new_features
                        
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
        
    def feature_fn(self, theta, *, model=None, features=None, as_numpy=False) -> jnp.ndarray:
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
        
        general_feature_fn = wrap(extract_features, model, self.frequency, as_numpy=as_numpy)
        feature_fn = lambda theta: general_feature_fn(theta, features)
        
        if self.jit_features:
            feature_fn_final = jax.jit(feature_fn)
        else:
            feature_fn_final = feature_fn
        return feature_fn_final(theta)
    
    @abstractmethod
    def run(self, *args, **kwargs) -> tuple[list[Model], SampleResults]:
        """Entry point for generating models."""
        pass
    
    @abstractmethod
    def _generate(self, N: int, d: int, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N new samples in the hypercube for d dimensions.
        
        Note that not all samplers support an arbitrary N.
        
        For adaptive samplers, `self.sampled_params` and `self.sampled_features` may be inspected.
        """
        raise NotImplementedError    