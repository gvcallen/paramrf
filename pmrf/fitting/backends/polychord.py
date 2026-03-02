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
    Bayesian fitter using the PolyChord nested sampling algorithm.
    
    This backend uses ``pypolychord`` to perform slice sampling, calculating 
    both the global evidence (logZ) and generating samples from the posterior 
    distribution.
    """
    def execute(
        self, 
        target: jnp.ndarray, 
        *, 
        fitted_params='maximum-likelihood', 
        nlive_factor=25, 
        **kwargs
    ) -> tuple[Model, Any]:
        """
        Execute the PolyChord nested sampling run.
        
        **NB:** This method should not be called directly. Call :meth:`run` instead.

        Parameters
        ----------
        target : jax.numpy.ndarray
            The extracted target features to fit against.
        fitted_params : {'maximum-likelihood', 'mean'}, default='maximum-likelihood'
            How to select the final point estimates for the returned model's 
            parameters from the posterior samples.
        nlive_factor : int, default=25
            A multiplier to determine the number of live points. The total 
            number of live points (``nlive``) defaults to ``nlive_factor * num_params``.
        **kwargs
            Additional keyword arguments passed directly to ``pypolychord.run``.

        Returns
        -------
        tuple[Model, Any]
            The fitted model (with parameter groups updated to the full posterior) 
            and the raw ``anesthetic.NestedSamples`` object.
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
            return float(self.log_likelihood(x, target))

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
        Encode anesthetic NestedSamples into a CSV string for serialization.
        """
        samples = results
        csv_str: str = samples.to_csv()
        stream.write(csv_str.encode('utf-8'))

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        """
        Reconstruct anesthetic NestedSamples from a serialized CSV string.
        """
        from anesthetic import NestedSamples, read_csv
        
        csv_str = stream.read().decode('utf-8')
        samples = NestedSamples(read_csv(io.StringIO(csv_str)))
        return samples