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
    def run(self, best_param_method='maximum-likelihood', **kwargs) -> AnestheticResults:
        # Dynamic imports
        import numpy as np
        import pypolychord

        # Get the model parameters
        param_names = [param.name for param in self._flat_params()]
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[name, f'\\theta_{{{name_replaced}}}'] for name, name_replaced in zip(param_names, dot_param_names)])
        
        # Generate prior and likelihood functions
        recon_fn, x0 = self._make_reconstruct_function(return_params=True, numpy_input=True)
        loglikelihood_fn = self._make_loglikelihood_function(numpy_input=True)
        prior_fn = self._make_prior_transform_function(numpy_input=True)
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
        
        for i, param_name in enumerate(param_names[0:-1]):
            if best_param_method == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
                
        return AnestheticResults(
            model=recon_fn(x0),
            initial_model=self.initial_model,
            frequency=self.model_frequency,
            measured=self.measured,
            features=self.feature_list,
            logger=self.logger,
            solver_results=nested_samples,
            solver_args=(),
            solver_kwargs=kwargs,
            fit_kwargs={'best_param_method': best_param_method}
        )    