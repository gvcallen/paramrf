from typing import Any
import jax
import jax.numpy as jnp
import io   
import h5py
import numpy as np

from pmrf._features import make_feature_function
from pmrf.fitting._bayesian import BayesianFitter, BayesianResults
from pmrf._util import time_string

def norm_logpdf(x, loc=0.0, scale=1.0):
    return -0.5 * jnp.log(2 * jnp.pi * scale**2) - 0.5 * ((x - loc)**2) / (scale**2)

def gaussian_log_likelihood(y_meas, y_model, sigma):
    return jnp.sum(norm_logpdf(jnp.real(y_meas), jnp.real(y_model), sigma))

class PolychordResults(BayesianResults):
    def encode_solver_results(self, group: h5py.Group):
        samples = self.solver_results
        group['samples'] = samples.to_csv()
        
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        from anesthetic import NestedSamples, read_csv
        
        csv_str = group['samples'][()]
        csv_str = csv_str.decode('utf-8') if isinstance(csv_str, bytes) else csv_str
        samples = NestedSamples(read_csv(io.StringIO(csv_str)))
        return samples
    
    def plot_params(self, param_names=None, title='params', label='posterior', priors=False, fig_size=None, fig=None, ax=None, **kwargs):
        from anesthetic import make_2d_axes
        
        nested_samples = self.solver_results
        params = param_names or list(self.model.params().keys())

        if ax is None:
            fig, ax = make_2d_axes(params, figsize=fig_size)

        for i in range(ax.shape[0]):  # Loop over rows
            for j in range(ax.shape[1]):  # Loop over columns
                axi = ax.iloc[i, j]
                axi.set_ylabel(axi.get_ylabel(), rotation='horizontal')

        if priors:
            prior_samples = nested_samples.prior()
            prior_samples.plot_2d(ax, label='prior', **kwargs)
        
        nested_samples.plot_2d(ax, label=label, **kwargs)
        if priors:
            ax.iloc[-1, 0].legend(bbox_to_anchor=(len(ax)/2, len(ax)), loc='lower center', ncol=2)
        
        return fig, ax    

class PolychordFitter(BayesianFitter):
    def run(self, best_param_method='maximum-likelihood', **kwargs) -> PolychordResults:
        # Dynamic imports
        import numpy as np
        import pypolychord
        
        # Get the model parameters
        flat_params = self.initial_model.flat_params()
        param_names = [p.name for p in flat_params] + [k for k in self.likelihood_params.keys()]
        dot_param_names = [name.replace('_', '.') for name in param_names]
        labeled_param_names = np.array([[name, f'\\theta_{{{name_replaced}}}'] for name, name_replaced in zip(param_names, dot_param_names)])
        
        # Generate prior and likelihood functions
        self.logger.info("Compiling model and likelihood function...")
        feature_fn, x0, recon_fn = make_feature_function(self.initial_model, self.feature_list, self.model_frequency, flat=True, return_params=True, return_recon_fn=True)
        x0_with_likelihood = list(x0) + [self.likelihood_params['sigma'].prior.mean]
        def jax_likelihood(flat_params_with_sigma) -> jnp.ndarray:
            sigma = flat_params_with_sigma[-1]
            model_features = feature_fn(flat_params_with_sigma[0:-1])
            return gaussian_log_likelihood(self.measured_features, model_features, sigma)
        
        priors = [param.prior for param in self.initial_model.flat_params()] + [self.likelihood_params['sigma'].prior]
        if any(x is None for x in priors):
            raise Exception("Found free parameter without a prior")
        
        prior_fn = lambda hypercube: np.array([prior.icdf(hypercube[i]) for i, prior in enumerate(priors)])
        jax_likelihood = jax.jit(jax_likelihood)
        likelihood_fn = lambda x: float(jax_likelihood(jnp.array(x)))
        _logL = likelihood_fn(x0_with_likelihood)

        # Run polychord. Useful parameters to investigate may be "precision_criterion" and "synchronous"
        kwargs.update({
            'prior': prior_fn,
            'paramnames': labeled_param_names,
        })
        
        self.logger.info(f'Fitting for {len(param_names)} model parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        self.logger.info(f'PolyChord started at {time_string()}')
        
        dumper = lambda _live, _dead, _logweights, logZ, _logZerr: self.logger.info(f'time: {time_string()} (logZ = {logZ:.2f})')
        nested_samples = pypolychord.run(
            likelihood_fn,
            len(param_names),
            dumper=dumper,
            **kwargs
        )
        
        self.logger.info(f'PolyChord finished at {time_string()}')
        
        x0 = np.array(x0)
        for i, param_name in enumerate(param_names[0:-1]):
            if best_param_method == 'mean':
                x0[i] = nested_samples[param_name].mean()
            elif best_param_method == 'maximum-likelihood':
                idx = jnp.argmax(nested_samples.logL.values)
                x0[i] = nested_samples[param_name].values[idx]
            else:
                self.logger.warning("Unknown best parameter method. Skipping")
                
        return PolychordResults(
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