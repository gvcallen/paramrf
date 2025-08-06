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

class dyPolychordFitter(BayesianFitter):
    def run(self, best_param_method='maximum-likelihood', nlive_init_factor=None, nlive_factor=None, **kwargs) -> AnestheticResults:
        # Dynamic imports
        import numpy as np
        import dyPolyChord.pypolychord_utils
        import dyPolyChord
        from anesthetic import read_chains
        
        nlive_init_factor = nlive_init_factor if nlive_init_factor is not None else 1
        nlive_factor = nlive_factor if nlive_factor is not None else 25
        
        num_params = self.model.num_flat_params + len(self.likelihood_params)
        param_names = self._flat_param_names()
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[name, f'\\theta_{{{name_replaced}}}'] for name, name_replaced in zip(param_names, dot_param_names)])
        
        settings_dict_in = {}
        dynamic_goal = 1.0
        
        kwargs.setdefault('ninit', nlive_init_factor * num_params)
        kwargs.setdefault('nlive_const', nlive_factor * num_params)
        
        if kwargs['nlive_const'] <= kwargs['ninit']:
            raise Exception('Number of dynamic live points must be greater than number of init live points')
        
        # kwargs.setdefault('paramnames', labeled_param_names)
        _base_dir = kwargs.pop('base_dir', 'chains')
        # settings_dict_in.setdefault('base_dir', kwargs.pop('base_dir', 'chains'))
        settings_dict_in.setdefault('file_root', 'test')
        
        # Generate prior and likelihood functions
        x0 = np.array(self.model.flat_params())
        loglikelihood_fn = self._make_log_likelihood_fn(as_numpy=True)
        prior_fn = self._make_prior_transform_fn(as_numpy=True)
        dumper = lambda _live, _dead, _logweights, logZ, _logZerr: self.logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')

        self.logger.info(f'Fitting for {len(param_names)} parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        self.logger.info(f'PolyChord started at {time_string()}')
        
        # Make a callable for running PolyChord
        combined_fn = dyPolyChord.pypolychord_utils.RunPyPolyChord(
            loglikelihood_fn, prior_fn, len(param_names),
        )

        # Run dyPolyChord
        # TODO clean up this parameter passing and settings dict stuff
        dyPolyChord.run_dypolychord(
            combined_fn,
            settings_dict_in=settings_dict_in,
            dynamic_goal=dynamic_goal,
            **kwargs,
        )
        
        chains_root, file_root = kwargs['chains'], kwargs['file_root']
        nested_samples = read_chains(f'{chains_root}/{file_root}')
        
        self.logger.info(f'dyPolyChord finished at {time_string()}')
        
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