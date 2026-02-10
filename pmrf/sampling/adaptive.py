from abc import abstractmethod

import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model

from pmrf.constants import FeatureInputT
from pmrf.frequency import Frequency
from pmrf._util import LivePlotter

class AdaptiveSampler(BaseSampler):
    def __init__(
        self,
        model: Model,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        initial_models: list[Model] | int = 10,
    ):    
        self.initial_models = list(initial_models) if not isinstance(initial_models, int) else initial_models
        super().__init__(model, frequency=frequency, features=features)
        
    def run(self, N: int | None = None, max_iterations: int = 100, num_success: int = 1, key=None, plot=None, jit_feature_fn=False, **kwargs) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('Key needed for AdaptiveSampler')
            
        if isinstance(self.initial_models, int):
            initial_key, key = jr.split(key)
            from pmrf.sampling._algos.latin_hypercube import LatinHypercubeSampler
            self.initial_models = LatinHypercubeSampler(self.model).run(self.initial_models, key=initial_key)[0]
        
        models = self.initial_models
        param_names = self.model.flat_param_names()
        
        theta_current = [model.flat_param_values() for model in models]
        U_current = [self.cumulative_distribution_fn(model.flat_param_values()) for model in models]
        features = []
        
        # # Initialize the plotter
        # plotter = LivePlotter(title="Function Samples", xlabel=f"Frequency ({self.frequency.unit})", ylabel="f(x)")

        # x_axis = np.linspace(0, 10, 100)

        # # Simulate 50 frames of animation
        # for t in range(50):
        #     # Create a sine wave that shifts over time
        #     y_wave = np.sin(x_axis + t/5.0)
            
        #     # Create a 'noise' line that dampens over time
        #     y_noise = np.random.normal(0, 1.0/(t+1), size=len(x_axis))
            
        #     # Update the curves completely
        #     plotter.update_curve("Moving Wave", y_wave, x_values=x_axis)
        #     plotter.update_curve("Dampening Noise", y_noise, x_values=x_axis)
            
        #     time.sleep(0.1)        
        
        sample_idx = 0
        def compute_sample(theta: jnp.ndarray):
            nonlocal sample_idx
            self.logger.info(f"Computing Sample #{sample_idx + 1} with {dict(zip(param_names, theta.tolist()))}")
            features_i = self.feature_fn(theta, jit=jit_feature_fn)
            sample_idx += 1
            return features_i
            
        self.logger.info('Computing initial sample outputs...')
        for theta in theta_current:
            features.append(compute_sample(theta))
        
        iteration = 0
        d = self.model.num_flat_params
        i_success = 0
        while iteration < max_iterations:
            U_next = self._generate(1, d, jnp.array(U_current), jnp.array(features), key=key, **kwargs)
            if U_next is None:
                i_success += 1
                if i_success >= num_success:
                    break
                else:
                    continue
                
            i_success = 0
            
            U_current.extend([u_next for u_next in U_next])
            for u in U_next:
                theta = self.inverse_cumulative_distribution_fn(u)
                features.append(compute_sample(theta))
                theta_current.append(theta)
            
            if N is not None and len(theta_current) >= N:
                break
            iteration += 1
            
        if iteration == max_iterations:
            self.logger.warning("Maximum iterations were reached during adaptive sampling")

        models = [self.model.with_params(theta) for theta in theta_current]
        results = SampleResults(initial_model=self.model, sampled_models=models, sampled_features=jnp.array(features))
        return models, results
    
    @abstractmethod
    def _generate(self, N: int, d: int, samples: jnp.ndarray, features: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for d dimensions.
        
        Note that not all samplers support an arbitrary N.
        
        Previous hypercube samples are passed in ``u`` of shape (M x d), alongside their corresponding ``features`` of shape (N x ...).
        New samples of shape (N x d) must be returned.
        """
        raise NotImplementedError