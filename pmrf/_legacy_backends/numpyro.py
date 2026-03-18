import io
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pmrf.fitting.bayesian import BayesianFitter
from pmrf.core import Model



class NumPyroFitter(BayesianFitter):
    r"""
    Base class for fitters utilizing the NumPyro probabilistic programming library.
    
    This base class handles the translation of the unified ParamRF model definitions 
    (priors and compiled likelihoods) into a dynamic NumPyro trace graph. It also 
    provides optimized binary serialization for the resulting MCMC/NS sample chains.

    .. rubric:: Methods

    .. autosummary::
       :nosignatures:
       
       make_numpyro_model
    """
    def make_numpyro_model(self, targets: jnp.ndarray):
        r"""
        Construct a dynamic NumPyro model function that mirrors the BayesianFitter logic.
        
        This method traces the prior distributions for both the physical model 
        parameters and the likelihood noise parameters using ``numpyro.sample``. 
        It then concatenates them and injects the lazily compiled ParamRF 
        log-likelihood graph directly into the trace as a ``numpyro.factor`` node.

        Parameters
        ----------
        targets : jax.numpy.ndarray
            The extracted measurement data to condition the likelihood on.

        Returns
        -------
        callable
            A dynamic parameter-less function representing the complete NumPyro 
            probabilistic model, ready to be passed to a sampling kernel.
        """
        import numpyro
        
        # 1. Trigger lazy compilation to ensure the unified log-likelihood graph is ready
        dummy_theta = jnp.zeros(self.num_params)
        _ = self.log_likelihood(dummy_theta, targets)
        
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
            numpyro.factor("log_likelihood", ll_fn(theta_combined, targets))
            
        return numpyro_model

    # --------------------------------------------------------------------------
    # Custom Binary Stream Serialization for NumPyro Samples
    # --------------------------------------------------------------------------
    @staticmethod
    def write_results(stream: io.BytesIO, results: Any):
        r"""
        Save the NumPyro sample dictionary as compressed numpy arrays.
        
        This overrides the default JSON serialization, providing significantly 
        faster write times and smaller file sizes for large MCMC traces.
        """
        np.savez_compressed(stream, **results)

    @staticmethod
    def read_results(stream: io.BytesIO) -> Any:
        r"""
        Reconstruct the NumPyro sample dictionary from the binary stream.
        """
        with np.load(stream) as data:
            return {k: data[k] for k in data.files}




class NumPyroMCMCFitter(NumPyroFitter):
    r"""
    Bayesian fitter using NumPyro Markov Chain Monte Carlo (MCMC) sampling.
    
    This backend utilizes ``numpyro.infer.MCMC``. By default, it operates using 
    the No-U-Turn Sampler (NUTS) kernel, which is highly efficient for continuous, 
    differentiable parameter spaces.
    """        
    def execute(
        self, 
        targets: jnp.ndarray, 
        *, 
        output_path: str | None = None,
        output_root: str | None = None,        
        kernel=None, 
        seed: int = 42, 
        fitted_params: str = 'mean', 
        **kwargs
    ) -> tuple[Model, Any]:
        r"""
        Execute the MCMC sampling run.
        
        **NB:** This method should not be called directly. Call :meth:`run` instead.

        Parameters
        ----------
        targets : jax.numpy.ndarray
            The extracted target features to fit against.
        output_path : str, optional
            Base directory for output traces (consumed by the base run method).
        output_root : str, optional
            Prefix for output files.
        kernel : numpyro.infer.mcmc.MCMCKernel, optional
            The specific MCMC transition kernel to use. If ``None``, defaults 
            to ``numpyro.infer.NUTS``.
        seed : int, default=42
            The PRNG key seed for JAX's random number generator.
        fitted_params : {'mean'}, default='mean'
            How to select the final point estimates for the returned model's 
            parameters from the posterior samples.
        **kwargs
            Additional keyword arguments passed directly to ``numpyro.infer.MCMC`` 
            (e.g., ``num_warmup``, ``num_samples``).

        Returns
        -------
        tuple[:class:`~pmrf.model.Model`, dict[str, jax.numpy.ndarray]]
            The fitted model and a dictionary containing the raw MCMC samples.
        """
        from numpyro.infer import MCMC, NUTS
        
        if kernel is None:
            kernel = NUTS
            
        param_names = self.model.flat_param_names()
        numpyro_model = self.make_numpyro_model(targets)
        
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
    r"""
    Bayesian fitter using NumPyro Nested Sampling.
    
    This backend utilizes ``numpyro.contrib.nested_sampling.NestedSampler`` to 
    compute both the posterior samples and the log-evidence of the model.
    """
    def execute(
        self, 
        targets: jnp.ndarray, 
        *, 
        constructor_kwargs=None, 
        terminated_kwargs=None, 
        seed: int = 42, 
        fitted_params: str = 'mean', 
        **kwargs
    ) -> tuple[Model, Any]:
        r"""
        Execute the Nested Sampling run.
        
        **NB:** This method should not be called directly. Call :meth:`run` instead.

        Parameters
        ----------
        targets : jax.numpy.ndarray
            The extracted target features to fit against.
        constructor_kwargs : dict, optional
            Keyword arguments passed directly to the ``NestedSampler`` constructor 
            (e.g., ``num_live_points``).
        terminated_kwargs : dict, optional
            Keyword arguments passed to control the termination criteria of the 
            sampling loop.
        seed : int, default=42
            The PRNG key seed for JAX's random number generator.
        fitted_params : {'mean'}, default='mean'
            How to select the final point estimates for the returned model's 
            parameters from the posterior samples.
        **kwargs
            Additional arguments. ``num_samples`` is extracted to determine 
            how many samples to draw from the final nested sampling weights.

        Returns
        -------
        tuple[:class:`~pmrf.model.Model`, dict[str, jax.numpy.ndarray]]
            The fitted model and a dictionary containing the generated samples.
        """
        from numpyro.contrib.nested_sampling import NestedSampler
        
        param_names = self.model.flat_param_names()
        numpyro_model = self.make_numpyro_model(targets)
        
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