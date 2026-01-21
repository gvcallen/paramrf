from typing import Iterator, TypeVar
from abc import ABC, abstractmethod
import logging

import numpy as np
import skrf
import jax
from jax import flatten_util
import jax.numpy as jnp
import equinox as eqx

from pmrf import extract_features, wrap
from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.constants import FeatureInputT

ModelT = TypeVar('ModelT', bound='Model')

class BaseSampler(ABC):
    """
    Abstract base class for parameter sampling engines.

    This class provides the infrastructure to generate random variations of a
    `pmrf.Model` based on the prior distributions of its parameters. It supports
    generating batches of models, batches of features (e.g., S-parameters),
    or iterating over generated models.

    Subclasses must implement `_generate_hypercube_samples` to define the
    sampling strategy (e.g., Monte Carlo, Latin Hypercube).

    Attributes
    ----------
    model : Model
        The base model defining the parameter structure and priors.
    N : int or None
        The number of samples for the current iteration context.
    logger : logging.Logger
        Logger instance.
    """
    def __init__(self, model: ModelT):
        """
        Initialize the sampler.

        Parameters
        ----------
        model : Model
            The model to sample from.
        """
        self.model: Model = model
        self.N = None
        self.logger = logging.getLogger(__name__)

    def __iter__(self) -> Iterator[Model]:
        """
        Iterate over generated models.

        Requires `range(N)` to have been called previously to set the sample count.

        Yields
        ------
        Model
            A new model instance with sampled parameters.

        Raises
        ------
        Exception
            If `range()` was not called before iteration.
        """
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
        """
        Return the number of samples in the current iteration context.

        Returns
        -------
        int
            Number of samples.

        Raises
        ------
        Exception
            If `range()` was not called before checking length.
        """
        if self.N is None:
            raise Exception('Error: to use this class as an iterator, call e.g. enumerate (CircuitSampler.range(n))')
        return self.N

    def range(self, N) -> ModelT:
        """
        Configure the sampler for iteration over `N` samples.

        Allows the Sampler to be used as an iterable.

        Examples
        --------
        >>> for i, system in enumerate(sampler.range(10)):
        ...     print(system)

        Parameters
        ----------
        N : int
            The number of samples to generate.

        Returns
        -------
        BaseSampler
            Returns self to allow chaining into an iterator.
        """
        self.N = N
        return self

    def generate_models(self, N) -> list[ModelT]:
        """
        Generate `N` random model instances using the sampler's engine.

        Parameters
        ----------
        N : int
            The number of samples to generate.

        Returns
        -------
        list[Model]
            A list of model instances with parameters sampled from the prior.
        """
        params_matrix = self._generate_params(N)

        models = []
        params, static = self.model._partition()
        _, ravel_fn = flatten_util.ravel_pytree(params)
        for i in range(N):
            models.append(eqx.combine(ravel_fn(params_matrix[i,:]), static))
        return models

    def generate_features(self, N, features: FeatureInputT, frequency: Frequency | skrf.Frequency, dont_jit=False, **kwargs) -> jnp.array:
        """
        Generate feature vectors for `N` random samples.

        This method vectorizes the feature extraction process using JAX for efficiency.

        Parameters
        ----------
        N : int
            The number of samples to generate.
        features : FeatureInputT
            The list or dictionary of features to extract (e.g., `['s11_db']`).
        frequency : Frequency or skrf.Frequency
            The frequency grid to evaluate the features on.
        dont_jit : bool, optional, default=False
            If True, disables JIT compilation for the vectorized function.
        **kwargs
            Additional arguments passed to the feature extractor.

        Returns
        -------
        jnp.array
            The array of generated features with shape `(N, n_features)`.
        """
        if isinstance(frequency, skrf.Frequency):
            frequency = Frequency.from_skrf(frequency)
        
        params_matrix = jnp.array(self._generate_params(N))
        params = self.model.flat_param_values()

        feature_fn = wrap(extract_features, self.model, frequency, features)
        
        self.logger.info('Compiling feature function...')
        _features0 = feature_fn(params)

        self.logger.info('Generating features.')
        
        vectorized_fn = jax.vmap(feature_fn)
        if not dont_jit:
            vectorized_fn = jax.jit(vectorized_fn)

        return vectorized_fn(params_matrix)
    
    def _generate_params(self, N):
        """
        Internal method to generate parameter values in physical space.

        It first generates samples in the unit hypercube [0, 1]^D and then
        maps them to physical space using the model's prior inverse CDF (PPF).

        Parameters
        ----------
        N : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Matrix of parameters with shape `(N, D)`.
        """
        params = self.model.flat_param_values()
        D = len(params)

        U = self._generate_hypercube_samples(N, D)
        prior = self.model.distribution()
        
        X = []
        for i in range(N):
            u = U[i, :]
            x = prior.icdf(u)
            X.append(x)

        return np.stack(X, axis=0)
    
    @abstractmethod
    def _generate_hypercube_samples(self, N, D) -> jnp.ndarray:
        """
        Generate samples in the unit hypercube [0, 1]^D.

        Parameters
        ----------
        N : int
            Number of samples.
        D : int
            Dimensionality (number of parameters).

        Returns
        -------
        jnp.ndarray
            Samples with shape `(N, D)`.
        """
        pass