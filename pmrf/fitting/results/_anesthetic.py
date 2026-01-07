from typing import Any
import io

import h5py
import jax.numpy as jnp

from pmrf.fitting._bayesian import BayesianResults

class AnestheticResults(BayesianResults):
    from anesthetic import NestedSamples

    @property
    def nested_samples(self) -> NestedSamples:
        return self.solver_results
    
    @property
    def sample_param_names(self) -> list[str]:
        columns = self.nested_samples.columns
        param_names = [columns[i][0] for i in range(len(columns))]
        param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]
        return param_names    

    def plot_params(self, param_names=None, prior=False, *args, **kwargs):
        from anesthetic import make_2d_axes
        import matplotlib.pyplot as plt

        param_names = param_names or self.sample_param_names
        fig, axes = make_2d_axes(param_names, *args, **kwargs)
        if prior:
            self.nested_samples.prior().plot_2d(axes, color='grey', alpha=0.5)
        return self.nested_samples.plot_2d(axes)    
    
    def prior_samples(self, equal_weights=False) -> jnp.ndarray:
        if equal_weights:
            nested_samples = self.nested_samples.prior_points()
        else:
            nested_samples = self.nested_samples.prior()
        samples = nested_samples.loc[:, self.sample_param_names].to_numpy()
        return jnp.array(samples)
    
    def posterior_samples(self, equal_weights=False) -> jnp.ndarray:
        if equal_weights:
            nested_samples = self.nested_samples.posterior_points()
        else:
            nested_samples = self.nested_samples
        prior_samples = nested_samples.loc[:, self.sample_param_names].to_numpy()
        return jnp.array(prior_samples)

    def weighted_samples(self) -> jnp.ndarray:
        return jnp.array(self.nested_samples.loc[:, self.sample_param_names].to_numpy())
    
    def weights(self) -> jnp.ndarray:
        return jnp.array(self.nested_samples.get_weights())
    
    def prior_weights(self) -> jnp.ndarray:
        return jnp.array(self.nested_samples.prior().get_weights())
    
    def encode_solver_results(self, group: h5py.Group):
        samples = self.solver_results
        group['samples'] = samples.to_csv()
        
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        from anesthetic import NestedSamples, read_csv
        import pandas as pd
        
        csv_str = group['samples'][()]
        csv_str = csv_str.decode('utf-8') if isinstance(csv_str, bytes) else csv_str
        samples = NestedSamples(read_csv(io.StringIO(csv_str)))
        # samples = NestedSamples(pd.read_csv(io.StringIO(csv_str), index_col=0))
        return samples    