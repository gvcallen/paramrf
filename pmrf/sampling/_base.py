from typing import Iterator
from abc import ABC, abstractmethod
import logging

import numpy as np
import skrf
import jax
from jax import flatten_util
import jax.numpy as jnp
import equinox as eqx

from pmrf import extract_features, wrap, wrap_prior
from frequency import Frequency
from pmrf._model import Model
from constants import FeatureInputT

class BaseSampler(ABC):
    def __init__(self, model: Model):
        self.model = model
        self.N = None
        self.logger = logging.getLogger(__name__)

    def __iter__(self) -> Iterator[Model]:
        if self.N is None:
            raise Exception('Error: to use this class as an iterator, call e.g. enumerate (CircuitSampler.range(n))')

        N = self.N
        params_matrix = self._generate_params(N)
        params, static = self.model.named_params()
        _, ravel_fn = flatten_util.ravel_pytree(params)
        for i in N:
            yield eqx.combine(ravel_fn(params_matrix[i,:]), static)

        self.N = None

    def __len__(self) -> int:
        if self.N is None:
            raise Exception('Error: to use this class as an iterator, call e.g. enumerate (CircuitSampler.range(n))')
        return self.N

    def range(self, N) -> Model:
        """Allows the CircuitSampler to be used as an iterable. To use, call e.g:
            for i, system in enumerate(sampler.range(10)).

        Args:
            n (int): The number of samples to generate

        Returns:
            Model: self
        """
        self.N = N
        return self

    def generate_models(self, N) -> list[Model]:
        """Generates N random models using the sampler's engine.

        Note that, if you want to generate samples one-by-one,
        you can use this class in iterator mode by passing N to the constructor
        or by using `Sampler.range(..)`.

        Args:
            n (int, optional): The number of samples to generate. Defaults to 10.

        Returns:
            _type_: Model | None
        """
        params_matrix = self._generate_params(N)

        models = []
        params, static = self.model.partition()
        _, ravel_fn = flatten_util.ravel_pytree(params)
        for i in range(N):
            models.append(eqx.combine(ravel_fn(params_matrix[i,:]), static))
        return models

    def generate_features(self, N, features: FeatureInputT, frequency: Frequency | skrf.Frequency, dont_jit=False, **kwargs) -> jnp.array:
        if isinstance(frequency, skrf.Frequency):
            frequency = Frequency.from_skrf(frequency)
        
        params_matrix = jnp.array(self._generate_params(N))
        params = self.model.flat_params()

        feature_fn = wrap(extract_features, self.model, frequency, features)
        
        self.logger.info('Compiling feature function...')
        _features0 = feature_fn(params)

        self.logger.info('Generating features.')
        
        vectorized_fn = jax.vmap(feature_fn)
        if not dont_jit:
            vectorized_fn = jax.jit(vectorized_fn)

        return vectorized_fn(params_matrix)
    
    def _generate_params(self, N):
        params = self.model.flat_params()
        D = len(params)

        U = self._generate_hypercube_samples(N, D)
        prior_transform_fn = wrap_prior(self.model)
        
        X = []
        for i in range(N):
            u = U[i, :]
            x = prior_transform_fn(u)
            X.append(x)

        return np.stack(X, axis=0)
    
    @abstractmethod
    def _generate_hypercube_samples(self, N, D) -> np.ndarray:
        pass