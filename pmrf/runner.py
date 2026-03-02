"""
The base runner class for running an algorithm such as fitting or sampling.
"""
from typing import Any
import io
from abc import ABC, abstractmethod
import logging

import jax
import jax.numpy as jnp
import jsonpickle

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.constants import FeatureSpecT
from pmrf.features import extract_features
from pmrf.util import RANK, LevelFilteredLogger


class BaseRunner(ABC):
    r"""
    The unified base class for all ParamRF runners (fitters and samplers).
    
    This class manages the core model, the frequency bands, and the specifications 
    for feature extraction. It also handles the lazy compilation of JAX-traced 
    functions for evaluating probabilities and model features.
    
    .. rubric:: Main methods

    .. autosummary::
       :nosignatures:
       
       cdf
       icdf
       log_prior
       model_features
       run
       write_results
       read_results

    Parameters
    ----------
    model : Model
        The base ParamRF model containing the free parameters to be evaluated.
    frequency : Frequency, optional
        The frequency definitions over which the model should be evaluated.
    features : FeatureSpecT, optional
        The target features to extract from the model's simulated network.
    **feature_kwargs
        Additional keyword arguments passed directly to the underlying 
        ``extract_features`` routine.

    Raises
    ------
    ValueError
        If the initialized ``model`` contains no free parameters.
    """
    def __init__(
        self,
        model: Model,
        *,
        frequency: Frequency | None = None,
        features: FeatureSpecT | None = None,
        **feature_kwargs,
    ):
        if model.num_flat_params == 0:
            raise ValueError("Model has no free parameters.")        
        
        self.model = model
        self.frequency = frequency
        self.features = features
        self.feature_kwargs = feature_kwargs
        self.output_path = None
        self.output_root = None

        if RANK == 0:
            self.logger = logging.getLogger(f"pmrf.{self.__class__.__name__}")
        else:
            self.logger = LevelFilteredLogger(null_level=logging.WARNING)

        self._cdf_fn = None
        self._icdf_fn = None
        self._log_prior_fn = None
        self._feature_fn = None

    def cdf(self, theta: jnp.ndarray) -> jnp.ndarray:
        r"""
        Evaluate the cumulative distribution function (CDF) for the model parameters.

        This computes the probability $P(X \leq \theta)$ and lazily compiles 
        the evaluation graph using ``jax.jit`` on the first call.

        Parameters
        ----------
        theta : jax.numpy.ndarray
            The parameter values to evaluate.

        Returns
        -------
        jax.numpy.ndarray
            The evaluated CDF probabilities, bounded between $0$ and $1$.
        """
        if self._cdf_fn is None:
            self._cdf_fn = jax.jit(self.model.distribution().cdf)
            
        return self._cdf_fn(jnp.array(theta))

    def icdf(self, u: jnp.ndarray) -> jnp.ndarray:
        r"""
        Evaluate the inverse cumulative distribution function (ICDF), or quantile function.

        This computes $\theta = F^{-1}(u)$ and lazily compiles the evaluation 
        graph using ``jax.jit`` on the first call.

        Parameters
        ----------
        u : jax.numpy.ndarray
            The probability values to evaluate, typically uniformly distributed 
            between $0$ and $1$.

        Returns
        -------
        jax.numpy.ndarray
            The corresponding parameter values $\theta$.
        """
        if self._icdf_fn is None:
            self._icdf_fn = jax.jit(self.model.distribution().icdf)
            
        return self._icdf_fn(jnp.array(u))
    
    def log_prior(self, theta: jnp.ndarray) -> float | jnp.ndarray:
        r"""
        Evaluate the log-prior probability $\log P(\theta)$ of the model parameters.
        
        The prior is calculated by summing the log-probabilities of the flat 
        parameter vector. The JAX graph is lazily compiled on the first call.

        Parameters
        ----------
        theta : jax.numpy.ndarray
            The parameter values to evaluate.

        Returns
        -------
        float or jax.numpy.ndarray
            The scalar log-prior probability.
        """
        if self._log_prior_fn is None:
            self.logger.debug("Lazily compiling model log-prior graph...")
            model_dist = self.model.distribution()
            
            @jax.jit
            def log_prior_fn(t):
                # Sum the log_probs of the flat parameter vector
                return jnp.sum(model_dist.log_prob(t))
                
            self._log_prior_fn = log_prior_fn
            
        return self._log_prior_fn(jnp.array(theta))    

    def model_features(self, theta: jnp.ndarray) -> jnp.ndarray:
        r"""
        Extract the RF features from the model for a given set of parameters.

        This function maps the parameters into the model, simulates it over the 
        defined frequency band, and extracts the target specifications. The 
        entire extraction pipeline is vectorized over the batch dimension and 
        lazily compiled via ``jax.jit(jax.vmap(...))``.

        Parameters
        ----------
        theta : jax.numpy.ndarray
            A 1D array of a single parameter set, or a 2D array representing 
            a batch of parameters.

        Returns
        -------
        jax.numpy.ndarray
            The extracted model features. Matches the batch dimension of ``theta``.

        Raises
        ------
        RuntimeError
            If ``frequency`` or ``features`` were not provided during initialization.
        """
        if self._feature_fn is None:
            if self.frequency is None or self.features is None:
                raise RuntimeError("Cannot compile feature function: frequency or features not set.")
            
            def _single_feature_fn(theta):
                m = self.model.with_params(theta)
                single_features = extract_features(m, self.frequency, self.features, **self.feature_kwargs)
                return single_features
            
            self._feature_fn = jax.jit(jax.vmap(_single_feature_fn))
        
        thetas_2d = jnp.atleast_2d(jnp.array(theta))
        features_2d = self._feature_fn(thetas_2d)           
        
        if theta.ndim == 2:
            return features_2d
        else:
            return features_2d[0]

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""
        Execute the primary algorithm for the runner.

        Must be implemented by subclasses.
        """
        pass
    
    @staticmethod
    def write_results(stream: io.BytesIO, results: Any):
        r"""
        Encode backend optimization or sampling results into a bytes stream.
        
        Parameters
        ----------
        stream : io.BytesIO
            The open byte stream to write to.
        results : Any
            The data structure containing the backend results.
            
        Notes
        -----
        The default implementation uses ``jsonpickle`` for robust, research-grade 
        serialization of complex Python objects.
        """
        pickle_str = jsonpickle.encode(results)
        stream.write(pickle_str.encode('utf-8'))

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        r"""
        Reconstruct backend optimization or sampling results from a bytes stream.
        
        Parameters
        ----------
        stream : io.BytesIO
            The open byte stream to read from.

        Returns
        -------
        Any
            The deserialized Python objects.
        """
        pickle_str = stream.read().decode('utf-8')
        return jsonpickle.decode(pickle_str)