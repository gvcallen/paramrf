from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model
from pmrf._util import LivePlotter

class OneshotSampler(BaseSampler):
    """Generates a fixed number of samples in one go."""
    def run(self, N: int, plot=None, key=None, jit_feature_fn=False) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('key needed for OneshotSampler')
        
        if plot is None:
            plot = []
              
        param_names = self.model.flat_param_names()
        u_samples = self._generate(N, self.model.num_flat_params, key=key)
        theta = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(u_samples)
        models = [self.model.with_params(params_i) for params_i in theta]
               
        # Initialize the plotter list
        plotters: list[LivePlotter] = []
        
        sample_idx = 0
        def compute_sample(theta: jnp.ndarray):
            nonlocal sample_idx
            
            params = dict(zip(param_names, theta.tolist()))
            self.logger.info(f"Computing Sample #{sample_idx + 1} with {params}")
            features_i = self.feature_fn(theta, jit=jit_feature_fn)
            sample_idx += 1
            
            # Create plotters lazily
            if len(plot) != 0 and len(plotters) == 0:
                for p in plot:
                    for f in range(features_i.shape[-1]):
                        plotters.append(LivePlotter(title=f"Sample Feature #{f}", xlabel=f"Frequency ({self.frequency.unit})", ylabel="Samples"))
            
            for f, (plotter, comp) in enumerate(zip(plotters, plot)):
                y = features_i[..., f]
                if comp == 'db':
                    y = 20*jnp.log10(jnp.abs(y))
                else:
                    raise Exception(f'{comp} component not supported yet in AdaptiveSampler')
                
                plotter.add_curve(f"#{sample_idx}, {params}", y, x_values=self.frequency.f_scaled)
                plotter.ax.set_title(f"Sample Feature #{f}, num_samples = {sample_idx + 1}")
            
            return features_i
            
        if self.features is not None:
            features = []
            for theta_i in theta:
                features.append(compute_sample(theta_i))
            features = jnp.array(features)
        else:
            features = None
        
        results = SampleResults(
            initial_model=self.model,
            sampled_models=models,
            sampled_params=theta,
            sampled_features=features,
        )
        
        return models, results

    @abstractmethod
    def _generate(self, N: int, d: int, key=None) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for D dimensions.
        """
        pass