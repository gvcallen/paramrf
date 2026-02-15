from abc import abstractmethod

from datetime import datetime
import jax.random as jr
import jax.numpy as jnp
import numpy as np

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
        **kwargs,
    ):    
        self.initial_models = list(initial_models) if not isinstance(initial_models, int) else initial_models
        super().__init__(model, frequency=frequency, features=features, **kwargs)
        
    def run(self, N: int | None = None, max_iterations: int = 100, num_success: int = 1, key=None, plot=None, jit_feature_fn=False, **kwargs) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('Key needed for AdaptiveSampler')
        if plot is None:
            plot = []
            
        if isinstance(self.initial_models, int):
            initial_key, key = jr.split(key)
            from pmrf.sampling._algos.latin_hypercube import LatinHypercubeSampler
            self.initial_models = LatinHypercubeSampler(self.model).run(self.initial_models, key=initial_key)[0]
        
        models = self.initial_models
        param_names = self.model.flat_param_names()
        
        theta = [model.flat_param_values() for model in models]
        U_current = [self.cumulative_distribution_fn(model.flat_param_values()) for model in models]
        features = []
        
        # Initialize the plotter list
        plotters: list[LivePlotter] = []
        
        sample_idx = 0
        def compute_sample(theta_i: jnp.ndarray):
            nonlocal sample_idx
            
            params = dict(zip(param_names, theta_i.tolist()))
            now = datetime.now()
            self.logger.info(f"Computing Sample #{sample_idx + 1} with {params} (time = {now.strftime("%Y-%m-%d %H:%M:%S")})")
            features_i = self.feature_fn(theta_i, jit=jit_feature_fn)
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
                        
            np.save(f"{self.output_path}/features.npy", features)
            np.save(f"{self.output_path}/theta.npy", np.array(theta))
            return features_i
            
        for theta_i in theta:
            features.append(compute_sample(theta_i))
        
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
                theta_i = self.inverse_cumulative_distribution_fn(u)
                features.append(compute_sample(theta_i))
                theta.append(theta_i)
            
            if N is not None and len(theta) >= N:
                break
            iteration += 1
            
        if iteration == max_iterations:
            self.logger.warning("Maximum iterations were reached during adaptive sampling")

        models = [self.model.with_params(theta_i) for theta_i in theta]
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