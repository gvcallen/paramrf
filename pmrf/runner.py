from typing import Any
import io
from abc import ABC, abstractmethod
import logging
import os

import jax
import jax.numpy as jnp
import jsonpickle

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.constants import FeatureInputT
from pmrf.features import extract_features
from pmrf.util import RANK, LevelFilteredLogger

class BaseRunner(ABC):
    """
    The unified base class for all ParamRF runners (fitters and samplers).
    """
    def __init__(
        self,
        model: Model,
        *,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        output_path: str | None = None,
        output_root: str | None = None,
        sparam_kind: str | None = None,
    ):
        self.model = model
        self.frequency = frequency
        self.features = features
        self.output_path = output_path
        self.output_root = output_root
        self.sparam_kind = sparam_kind

        if RANK == 0:
            self.logger = logging.getLogger(f"pmrf.{self.__class__.__name__}")
        else:
            self.logger = LevelFilteredLogger(null_level=logging.WARNING)

        if self.output_path is not None and RANK == 0:
            os.makedirs(self.output_path, exist_ok=True)

        self._cdf_fn = None
        self._icdf_fn = None
        self._log_prior_fn = None
        self._feature_fn = None

    def cdf(self, theta: jnp.ndarray) -> jnp.ndarray:
        if self._cdf_fn is None:
            self._cdf_fn = jax.jit(self.model.distribution().cdf)
            
        return self._cdf_fn(jnp.array(theta))

    def icdf(self, u: jnp.ndarray) -> jnp.ndarray:
        if self._icdf_fn is None:
            self._icdf_fn = jax.jit(self.model.distribution().icdf)
            
        return self._icdf_fn(jnp.array(u))
    
    def log_prior(self, theta: jnp.ndarray) -> float | jnp.ndarray:
        """Lazily compiles and evaluates the log-prior probability of the model parameters."""
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
        if self._feature_fn is None:
            if self.frequency is None or self.features is None:
                raise RuntimeError("Cannot lazily compile features: frequency or features not set.")
            
            def _single_feature_fn(theta):
                m = self.model.with_params(theta)
                return extract_features(m, self.frequency, self.features, sparam_kind=self.sparam_kind)
            
            self._feature_fn = jax.jit(jax.vmap(_single_feature_fn))
        
        thetas_2d = jnp.atleast_2d(jnp.array(theta))
        features_2d = self._feature_fn(thetas_2d)
        
        if theta.ndim == 2:
            return features_2d
        else:
            return features_2d[0]

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
    
    @staticmethod
    def write_results(stream: io.BytesIO, results: Any):
        """
        Encodes backend results into a bytes stream.
        Default implementation uses jsonpickle for research-grade serialization.
        """
        pickle_str = jsonpickle.encode(results)
        stream.write(pickle_str.encode('utf-8'))

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        """
        Reconstructs backend results from a bytes stream.
        Default implementation uses jsonpickle.
        """
        pickle_str = stream.read().decode('utf-8')
        return jsonpickle.decode(pickle_str)