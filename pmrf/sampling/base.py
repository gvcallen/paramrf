from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from typing import Any

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
        features: FeatureInputT | None = 's',
    ):
        self.model: Model = model
        self.frequency = frequency
        self.features = features
        self.logger = logging.getLogger(__name__)
        
        params, self._static = self.model.partition()
        self._flat_params, self._ravel_fn = flatten_util.ravel_pytree(params)
        
    def inverse_cumulative_distribution_fn(self, u, as_numpy=False):
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
        num_model_params = self.model.num_flat_params
        
        @jax.jit
        def icdf_fn(u):
            return model_distribution.icdf(u)
            
        if as_numpy:
            icdf_transform_fn_jax = icdf_fn
            icdf_fn = lambda hypercube: np.array(icdf_transform_fn_jax(hypercube))
        
        # self.logger.info('Compiling inverse cumulative distribution function...')
        # _icdf_val = icdf_fn(jnp.array([0.5] * (num_model_params)))
        return icdf_fn(u)
    
    def cumulative_distribution_fn(self, theta, as_numpy=False):
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
        
        # self.logger.info('Compiling cumulative distribution function...')
        # _cdf_val = cdf_fn(self.model.flat_param_values())
        return cdf_fn(theta)
        
    def feature_fn(self, theta, as_numpy=False, jit=True):
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
        if self.frequency is None or self.features is None:
            raise Exception('Cannot make a feature function without a sampling frequency or features')
        
        general_feature_fn = wrap(extract_features, self.model, self.frequency, as_numpy=as_numpy)
        feature_fn = lambda theta: general_feature_fn(theta, self.features)
        
        if jit:
            feature_fn_final = jax.jit(feature_fn)
        else:
            feature_fn_final = feature_fn
        return feature_fn_final(theta)
    
    @abstractmethod
    def run(self, *args, **kwargs) -> tuple[list[Model], SampleResults]:
        """Entry point for generating models."""
        pass    