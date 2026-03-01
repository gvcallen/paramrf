import io
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pmrf.fitting.bayesian import BayesianFitter
from pmrf.models import Model

class NumPyroFitter(BayesianFitter):
    """
    Base class for fitters utilizing the NumPyro probabilistic programming library.
    """
    def make_numpyro_model(self, target_features: jnp.ndarray):
        """
        Constructs a dynamic NumPyro model function that perfectly mirrors the 
        compiled BayesianFitter logic.
        """
        import numpyro
        
        # 1. Trigger lazy compilation to ensure the unified log-likelihood graph is ready
        dummy_theta = jnp.zeros(self.num_params)
        _ = self.log_likelihood(dummy_theta, target_features)
        
        ll_fn = self._log_likelihood_fn
        
        def numpyro_model():
            # 2. Trace model parameter priors (Allows NumPyro to handle bound transforms)
            x_list = []
            for name, param in self.model.named_flat_params().items():
                x_list.append(numpyro.sample(name, param.distribution))
            x = jnp.stack(x_list)
            
            # 3. Trace likelihood parameter priors (e.g., sigma)
            lik_list = []
            for name, param in self.likelihood_params.items():
                lik_list.append(numpyro.sample(name, param.distribution))
            
            # 4. Concatenate trace values to match the unified Fitter signature
            if lik_list:
                lik_arr = jnp.stack(lik_list)
                theta_combined = jnp.concatenate([x, lik_arr])
            else:
                theta_combined = x
                
            # 5. Inject the compiled Fitter log-likelihood directly into the NumPyro trace!
            numpyro.factor("log_likelihood", ll_fn(theta_combined, target_features))
            
        return numpyro_model

    # --------------------------------------------------------------------------
    # Custom Binary Stream Serialization for NumPyro Samples
    # --------------------------------------------------------------------------
    @staticmethod
    def write_results(stream: io.BytesIO, results: Any):
        """
        Saves the NumPyro sample dictionary as compressed numpy arrays.
        This is significantly faster and smaller than jsonpickle for MCMC traces.
        """
        np.savez_compressed(stream, **results)

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        """
        Reconstructs the NumPyro sample dictionary from the binary stream.
        """
        with np.load(stream) as data:
            return {k: data[k] for k in data.files}


class NumPyroMCMCFitter(NumPyroFitter):
    """
    NumPyro MCMC: Markov Chain Monte Carlo (MCMC) sampling using ``numpyro.infer.MCMC``.
    
    Defaults to using the No-U-Turn Sampler (NUTS).
    """        
    def optimize(
        self, 
        target_features: jnp.ndarray, 
        *, 
        kernel=None, 
        seed: int = 42, 
        fitted_params: str = 'mean', 
        **kwargs
    ) -> tuple[Model, Any]:
        """Executes the MCMC sampling run."""
        from numpyro.infer import MCMC, NUTS
        
        if kernel is None:
            kernel = NUTS
            
        param_names = self.model.flat_param_names()
        numpyro_model = self.make_numpyro_model(target_features)
        
        self.logger.info(f'Fitting for {len(param_names)} model parameter(s) using MCMC...')
        
        kwargs.setdefault('num_warmup', 500)
        kwargs.setdefault('num_samples', 1000)
        
        mcmc = MCMC(kernel(numpyro_model), **kwargs)
        mcmc.run(jax.random.PRNGKey(seed))
        
        samples = mcmc.get_samples()

        # Posterior means
        x0 = np.array(self.model.flat_param_values())
        for i, param_name in enumerate(param_names):
            if fitted_params == 'mean':
                x0[i] = float(samples[param_name].mean())
                
        fitted_model = self.model.with_params(x0)
        
        return fitted_model, samples


class NumPyroNSFitter(NumPyroFitter):
    """
    NumPyro NS: Nested sampling using ``numpyro.contrib.nested_sampling.NestedSampler``.
    """
    def optimize(
        self, 
        target_features: jnp.ndarray, 
        *, 
        constructor_kwargs=None, 
        terminated_kwargs=None, 
        seed: int = 42, 
        fitted_params: str = 'mean', 
        **kwargs
    ) -> tuple[Model, Any]:
        """Executes the Nested Sampling run."""
        from numpyro.contrib.nested_sampling import NestedSampler
        
        param_names = self.model.flat_param_names()
        numpyro_model = self.make_numpyro_model(target_features)
        
        self.logger.info(f'Fitting for {len(param_names)} model parameter(s) using Nested Sampling...')
        
        constructor_kwargs = constructor_kwargs or {}
        terminated_kwargs = terminated_kwargs or {}
        
        ns = NestedSampler(numpyro_model, constructor_kwargs=constructor_kwargs, termination_kwargs=terminated_kwargs)
        ns.run(jax.random.PRNGKey(seed))
        
        rng_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
        samples = ns.get_samples(sample_key, num_samples=kwargs.get('num_samples', 1000))

        # Posterior means
        x0 = np.array(self.model.flat_param_values())
        for i, param_name in enumerate(param_names):
            if fitted_params == 'mean':
                x0[i] = float(samples[param_name].mean())
                
        fitted_model = self.model.with_params(x0)
        
        return fitted_model, samples