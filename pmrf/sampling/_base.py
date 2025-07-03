from typing import Iterator
from abc import ABC, abstractmethod
import logging

import numpy as np
import skrf
import jax
from jax import flatten_util
import jax.numpy as jnp
import equinox as eqx

from pmrf._frequency import Frequency
from pmrf._model import Model
from pmrf._features import extract_features, make_feature_function
from pmrf._constants import FeatureInputT

class BaseSampler(ABC):
    def __init__(self, model: Model):
        self.model = model
        self.N = None
        self.logger = logging.getLogger(__name__)

    def __iter__(self) -> Iterator[Model]:
        if self.N is None:
            raise Exception('Error: to use this class as an iterator, call e.g. enumerate (CircuitSampler.range(n))')

        N = self.N
        params_matrix = self._generate_param_matrix(N)
        params, static = self.model.params()
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
        params_matrix = self._generate_param_matrix(N)

        models = []
        params, static = self.model.partition()
        _, ravel_fn = flatten_util.ravel_pytree(params)
        for i in range(N):
            models.append(eqx.combine(ravel_fn(params_matrix[i,:]), static))
        return models

    def generate_features(self, N, features: FeatureInputT, frequency: Frequency | skrf.Frequency, dtype: jnp.dtype | np.dtype = jnp.complex128, dont_jit=False) -> jnp.array:
        if isinstance(frequency, skrf.Frequency):
            frequency = Frequency.from_skrf(frequency)
        
        params_matrix = jnp.array(self._generate_param_matrix(N))
        feature_fn, params_out = make_feature_function(self.model, features, frequency, dtype=dtype, jit=not dont_jit, flat=True, return_params=True)
        self.logger.info('Compiling feature function...')
        _features0 = feature_fn(params_out)

        self.logger.info('Generating features.')
        
        vectorized_fn = jax.vmap(feature_fn)
        if not dont_jit:
            vectorized_fn = jax.jit(vectorized_fn)

        return vectorized_fn(params_matrix)
    
    def _generate_param_matrix(self, N):
        params = self.model.flat_params()
        D = len(params)

        X = self._generate_hypercube_samples(N, D)    
        
        mapped = []
        for d in range(D):
            x_d = X[:, d]  # Shape (N,)
            p = params[d]
            if p.prior is not None:
                mapped_d = p.prior.icdf(x_d)
            else:
                mapped_d = p.min + x_d * p.max
            mapped.append(mapped_d)

        return np.stack(mapped, axis=0).T
    
    @abstractmethod
    def _generate_hypercube_samples(self, N, D) -> np.ndarray:
        pass