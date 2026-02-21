from abc import ABC, abstractmethod
import jax
import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model
from pmrf.constants import FeatureInputT
from pmrf.frequency import Frequency
from pmrf._util import lhs_sample

class AdaptiveSampler(BaseSampler, ABC):
    def __init__(
        self,
        model: Model,
        *,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        initial_models_factor: int | None = 10,
        initial_models: list[Model] | int | None = None,
        **kwargs,
    ):
        if initial_models_factor is not None and initial_models is not None:
            raise Exception("Cannot pass both initial_models_factor and initial_models to AdaptiveSampler")
        if initial_models_factor is not None:
            initial_models = initial_models_factor * model.num_flat_params
            
        if isinstance(initial_models, int) and initial_models < 2:
            raise Exception("Number of initial models must be at least 2")
        
        self.initial_models = list(initial_models) if not isinstance(initial_models, int) else initial_models

        super().__init__(model, frequency=frequency, features=features, **kwargs)

    def run(self, N: int | None = None, *, batch_size: int | None = 1, max_iterations: int | None = None, key=None, plot=None, **kwargs) -> tuple[list[Model], SampleResults]:
        if batch_size is None:
            batch_size = 1
        if key is None:
            raise Exception('Key needed for AdaptiveSampler')
        
        d = self.model.num_flat_params
        if isinstance(self.initial_models, int):
            key, initial_key = jr.split(key)
            initial_Us = lhs_sample(self.initial_models, d, key=initial_key)
            initial_thetas = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(initial_Us)
            self.initial_models = [self.model.with_params(theta) for theta in initial_thetas]
        
        initial_thetas = jnp.array([model.flat_param_values() for model in self.initial_models])
        num_initial_samples = len(initial_thetas)
        for i in range(0, num_initial_samples, batch_size):
            batch_theta = initial_thetas[i : i + batch_size]
            self.add_samples(batch_theta, plot=plot)
        
        iteration = 0
        while True:
            # We try ask for self.batch_size samples at a time, but may receive less
            key, generate_key = jr.split(key)
            U_next = self._generate(batch_size, d, key=generate_key, **kwargs)
            
            if U_next is None:
                self.logger.info("Sampling converged.")
                
            thetas = jnp.array([self.inverse_cumulative_distribution_fn(u) for u in U_next])
            num_samples = len(thetas)
            for i in range(0, num_samples, batch_size):
                batch_theta = thetas[i : i + batch_size]
                self.add_samples(batch_theta, plot=plot)            
            
            if N is not None and len(self.sampled_params) >= N:
                break
            iteration += 1
            if max_iterations is not None and iteration == max_iterations:
                break            
            
        if max_iterations is not None and iteration == max_iterations:
            self.logger.warning("Maximum iterations were reached during adaptive sampling")

        models = [self.model.with_params(theta) for theta in self.sampled_params]
        results = SampleResults(initial_model=self.model, sampled_models=models, sampled_params=self.sampled_params, sampled_features=self.sampled_features)
        return models, results

    @abstractmethod
    def _generate(self, N: int, d: int, *, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N new samples in the hypercube for d dimensions.
        
        It is not a requirement to return N samples: 1 <= n_samples < N may returned.
        To signify convergence, `None` may be returned.
        
        Note that `self.sampled_params` and `self.sampled_features` may be inspected at each iteration.
        """
        raise NotImplementedError