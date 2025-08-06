from typing import Any
import jax
import jax.numpy as jnp
import io   
import h5py
import numpy as np

from pmrf.fitting._bayesian import BayesianFitter, BayesianResults
from pmrf.fitting.results import AnestheticResults
from pmrf._util import time_string
   
PolychordResults = AnestheticResults

class PolychordFitter(BayesianFitter):
    def run(self, best_param_method='maximum-likelihood', nlive_factor=None, **kwargs) -> AnestheticResults:
        # Dynamic imports
        import numpy as np
        import pypolychord
        
        if not 'nlive' in kwargs and nlive_factor is not None:
            kwargs['nlive'] = nlive_factor * (self.model.num_flat_params + len(self.likelihood_params))
        
        # Get the model parameters
        param_names = self._flat_param_names()
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[name, f'\\theta_{{{name_replaced}}}'] for name, name_replaced in zip(param_names, dot_param_names)])
        
        # Generate prior and likelihood functions
        x0 = np.array(self.model.flat_params())
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
                
        fit_model = self.model.with_flat_params(x0)
                
        return AnestheticResults(
            fit_model=fit_model,
            initial_model=self.model,
            frequency=self.frequency,
            measured=self.measured,
            features=self.feature_list,
            logger=self.logger,
            solver_results=nested_samples,
            solver_args=(),
            solver_kwargs=kwargs,
            fit_kwargs={'best_param_method': best_param_method}
        )    