import io
from typing import Any
import numpy as np
import h5py
import jax.numpy as jnp

from pmrf.fitting.bayesian import BayesianFitter
from pmrf.models.model import Model
from pmrf.util import time_string

class PolyChordFitter(BayesianFitter):
    """
    PolyChord: Nested slice sampling backend using ``pypolychord``.
    
    PolyChord provides global evidence (logZ) and posterior samples.
    """
    def optimize(
        self, 
        target_features: jnp.ndarray, 
        *, 
        fitted_params='maximum-likelihood', 
        nlive_factor=25, 
        **kwargs
    ) -> tuple[Model, Any]:
        """
        Executes the PolyChord nested sampling run.
        """
        # Dynamic imports for heavy external dependencies
        import pypolychord
        from pmrf.parameters import ParameterGroup
        from pmrf.distributions import AnestheticDistribution
        
        # 1. Setup PolyChord Configuration
        kwargs.setdefault('nlive', nlive_factor * self.num_params)

        if self.output_path is not None:
            kwargs.setdefault('base_dir', f'{self.output_path}/chains')
            kwargs.setdefault('file_root', self.output_root or 'polychord')
        
        # 2. Map Parameter Names
        param_names = self.model.flat_param_names() + list(self.likelihood_params.keys())
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[n, dn] for n, dn in zip(param_names, dot_param_names)])
        
        # 3. Define Wrappers for Lazily Compiled JAX Functions
        def log_likelihood_np(x):
            return float(self.log_likelihood(x, target_features))

        def prior_np(u):
            return np.array(self.icdf(u))
        
        theta0 = 0.5 * jnp.ones(len(param_names))
        _ = log_likelihood_np(theta0)
        _ = prior_np(theta0)

        dumper = lambda _live, _dead, _logweights, logZ, _logZerr: \
            self.logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')

        # 4. Execute Nested Sampling
        self.logger.info(f'PolyChord started at {time_string()}')
        nested_samples = pypolychord.run(
            log_likelihood_np,
            len(param_names),
            dumper=dumper,
            prior=prior_np,
            paramnames=labeled_param_names,
            **kwargs
        )
        self.logger.info(f'PolyChord finished at {time_string()}')
        
        # 5. Determine Best Parameters
        x0 = np.array(self.model.flat_param_values())
        num_model_params = self.model.num_flat_params
        
        for i, param_name in enumerate(param_names[0:num_model_params]):
            if fitted_params == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif fitted_params == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
                
        # 6. Update Model Priors with the full Posterior Distribution
        model_param_names = self.model.flat_param_names()
        param_group = ParameterGroup(
            model_param_names, 
            AnestheticDistribution(nested_samples, model_param_names)
        )
        fitted_model = self.model.with_params(x0).with_param_groups(param_group)
        return fitted_model, nested_samples

    @staticmethod
    def write_results(stream: io.BytesIO, results: Any):
        """
        Encodes anesthetic NestedSamples into HDF5.
        We save the sample data as a dataset and use attributes for metadata like logZ.
        """
        samples = results
        csv_str: str = samples.to_csv()
        stream.write(csv_str.encode('utf-8'))

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        """
        Reconstructs anesthetic NestedSamples from HDF5.
        """
        from anesthetic import NestedSamples, read_csv
        
        csv_str = stream.read().decode('utf-8')
        samples = NestedSamples(read_csv(io.StringIO(csv_str)))
        return samples