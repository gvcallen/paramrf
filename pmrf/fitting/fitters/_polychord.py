from typing import Any
import jax
import jax.numpy as jnp
import io   
import h5py
import numpy as np

from pmrf.fitting._bayesian import BayesianFitter
from pmrf.fitting.results import AnestheticResults
from pmrf._util import time_string, explicit_kwargs
   
# For legacy imports
PolyChordResults = AnestheticResults

class PolyChordFitter(BayesianFitter):
    """
    Polychord fitter using pypolychord.run.
    
    Polychord has its own license available at https://github.com/PolyChord/PolyChordLite.
    """
    def run(self, best_param_method='maximum-likelihood', nlive_factor=None, **kwargs) -> AnestheticResults:
        # Dynamic imports
        import numpy as np
        import pypolychord
        
        if not 'nlive' in kwargs and nlive_factor is not None:
            kwargs['nlive'] = nlive_factor * self.num_params

        kwargs.setdefault('base_dir', f'{self.output_path}/chains')
        kwargs.setdefault('file_root', self.output_root)
        
        # Get the model parameters
        param_names = self._flat_param_names()
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[name, f'{name_replaced}'] for name, name_replaced in zip(param_names, dot_param_names)])
        
        # Generate prior and likelihood functions
        x0 = np.array(self.initial_model.flat_params())
        loglikelihood_fn = self._make_log_likelihood_fn(as_numpy=True)
        prior_fn = self._make_prior_transform_fn(as_numpy=True)
        dumper = lambda _live, _dead, _logweights, logZ, _logZerr: self.logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')

        self.logger.info(f'Fitting for {len(param_names)} parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        self.logger.info(f'PolyChord started at {time_string()}')
        nested_samples = pypolychord.run(
            loglikelihood_fn,
            len(param_names),
            dumper=dumper,
            prior=prior_fn,
            paramnames=labeled_param_names,
            **kwargs
        )
        
        self.logger.info(f'PolyChord finished at {time_string()}')
        
        for i, param_name in enumerate(param_names[0:-self.num_likelihood_params]):
            if best_param_method == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
                
        fitted_model = self.initial_model.with_flat_params(x0)
        
        # settings = self._settings(kwargs, explicit_kwargs())
        settings = self._settings(kwargs)
        self.results = AnestheticResults(
            measured=self.measured,
            initial_model=self.initial_model,
            fitted_model=fitted_model,
            solver_results=nested_samples,
            settings=settings,
        )

        return self.results