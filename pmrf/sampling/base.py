from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from typing import Any
import os
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.model import Model, wrap
from pmrf.constants import FeatureInputT, FeatureT
from pmrf.features import extract_features
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
        *,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        output_path: str | None = None,
        batch_size: int = 1,
    ):
        if model.num_flat_params == 0:
            raise Exception("Model has no free parameters to sample")
        
        self.model: Model = model
        self.frequency: Frequency = frequency
        self.features: FeatureInputT = features
        self.logger = logging.getLogger(__name__)
        self.output_path: str = output_path
        self.batch_size: int = batch_size
        
        self.sampled_params: jnp.ndarray = None
        self.sampled_features: jnp.ndarray = None
        self.feature_plotters: list[LivePlotter] = None
        self.convergence_plotter: LivePlotter = LivePlotter("Convergence", "Iteration", "Loss")
        
        self.logger.info(f"Sampling model with {model.num_flat_params} flat params: {self.model.flat_param_names()}")
        
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            
    def _compute_batch_features(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """
        Helper to compute features for a batch of parameters simultaneously.
        
        Parameters
        ----------
        thetas : jnp.ndarray
            Batch of parameters with shape (N, D).
            
        Returns
        -------
        jnp.ndarray
            Batch of features with shape (N, F).
        """
        # We reconstruct the extraction logic here to apply vmap efficiently.
        # This avoids the overhead of wrapping/jitting per sample.
        general_feature_fn = wrap(extract_features, self.model, self.frequency, as_numpy=False)
        
        def single_sample_fn(t):
            return general_feature_fn(t, self.features)
            
        # Use vmap to parallelize the feature extraction over the batch dimension
        batch_fn = jax.vmap(single_sample_fn)
        
        # Apply JIT if requested
        batch_fn = jax.jit(batch_fn)
            
        return batch_fn(thetas)

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

        # 1. Normalize Input: Ensure theta is 2D (N x D)
        is_single_input = theta.ndim == 1
        theta_batch = jnp.atleast_2d(theta)
        N, D = theta_batch.shape
        
        # 2. Check for existing samples to avoid re-computation
        # We use a mask to identify which of the incoming thetas are new
        if self.sampled_params is not None:
            # Broadcast check: (N, 1, D) == (1, M, D) -> (N, M)
            # Check for exact matches across parameter dimensions
            matches = jnp.all(jnp.equal(theta_batch[:, None, :], self.sampled_params[None, :, :]), axis=-1)
            is_present = jnp.any(matches, axis=1)
            # Get index in history for the ones that are present
            present_indices = jnp.argmax(matches, axis=1)
        else:
            is_present = jnp.zeros(N, dtype=bool)
            present_indices = jnp.zeros(N, dtype=int) # Dummy values

        # 3. Compute features for NEW samples only (Parallelized)
        new_indices_mask = ~is_present
        new_thetas = theta_batch[new_indices_mask]
        
        new_computed_features = None
        num_new_samples = len(new_thetas)
        if num_new_samples > 0 and self.features is not None:
            time_str = datetime.now().strftime("%H:%M:%S")
            
            num_existing_samples = len(self.sampled_params) if self.sampled_params is not None else 0
            if num_new_samples == 1:
                self.logger.info(f"Computing sample #{num_existing_samples + 1} at {time_str}")
            else:
                self.logger.info(f"Computing samples #{num_existing_samples + 1}-{num_existing_samples + num_new_samples} at {time_str}")
            for theta in new_thetas:
                printable_params = {k: round(float(v), 2) for k, v in zip(self.model.flat_param_names(), theta)}
                self.logger.info(f"Parameters = {printable_params}")
            
            # This is the parallelized call
            new_computed_features = self._compute_batch_features(new_thetas)

        # 4. Construct the full result array (combining found history + newly computed)
        # We need to assemble the results in the order of the input `theta`
        final_features = None
        if self.features is not None:
            # Create a container for the results
            # FIX: Use shape[1:] to capture (F, M) instead of just shape[1] which only captures (F,)
            if self.sampled_features is not None:
                ref_shape = self.sampled_features.shape[1:]
                dtype = self.sampled_features.dtype
            else:
                ref_shape = new_computed_features.shape[1:]
                dtype = new_computed_features.dtype
            
            # Combine batch size N with the reference feature dimensions
            feature_shape = (N,) + ref_shape
            final_features = jnp.zeros(feature_shape, dtype=dtype)

            # Fill in existing features
            if jnp.any(is_present):
                final_features = final_features.at[is_present].set(self.sampled_features[present_indices[is_present]])
            
            # Fill in new features
            if jnp.any(new_indices_mask):
                final_features = final_features.at[new_indices_mask].set(new_computed_features)
        # 5. Update State (Store new parameters and features)
        if len(new_thetas) > 0:
            if self.sampled_params is None:
                self.sampled_params = new_thetas
                self.sampled_features = new_computed_features
            else:
                self.sampled_params = jnp.vstack((self.sampled_params, new_thetas))
                if self.sampled_features is not None:
                    self.sampled_features = jnp.vstack((self.sampled_features, new_computed_features))
            
            # Save to disk
            if self.output_path is not None:
                np.save(f"{self.output_path}/params.npy", self.sampled_params)
                np.save(f"{self.output_path}/features.npy", self.sampled_features)

        # 6. Plotting
        # LivePlotter typically updates one curve at a time, so we loop through the batch.
        if plot is not None:
            # Create plotters (lazily)
            if self.feature_plotters is None:
                self.feature_plotters = []
                for p in plot:
                    self.feature_plotters.append(LivePlotter(title=f"{p}", xlabel=f"Frequency ({self.frequency.unit})", ylabel=f"{p}"))
            
            # We must re-compute features specifically for the 'plot' list if they differ from stored features
            # or if we just want to visualize the current batch. 
            # Note: Ideally we cache this too, but for now we compute to ensure correct plotting columns.
            
            # Helper for plotting features
            general_plot_fn = wrap(extract_features, self.model, self.frequency, as_numpy=False)
            plot_fn_mapped = jax.vmap(lambda t: general_plot_fn(t, plot))
            
            # Calculate for whole batch
            batch_plot_features = plot_fn_mapped(theta_batch)

            for batch_idx in range(N):
                current_theta = theta_batch[batch_idx]
                param_dict = dict(zip(self.model.flat_param_names(), current_theta.tolist()))
                
                for i, (plotter, feature_name) in enumerate(zip(self.feature_plotters, plot)):
                    feature = batch_plot_features[batch_idx, ..., i]
                    plotter.ax.set_title(f"{feature_name} (num_samples = {len(self.sampled_params)})")
                    plotter.add_curve(f"{param_dict}", y_values=jnp.real(feature), x_values=self.frequency.f_scaled)

        # Return result with original dimensionality (squeeze if it was single input)
        if final_features is not None and is_single_input:
            return final_features[0]
        return final_features
                        
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
        
        feature_fn_final = jax.jit(feature_fn)
        return feature_fn_final(theta)
    
    @abstractmethod
    def run(self, *args, **kwargs) -> tuple[list[Model], SampleResults]:
        """Entry point for generating models."""
        pass