import jax
import jax.numpy as jnp
from typing import Any
import h5py

from pmrf.fitting.bayesian import BayesianFitter, BayesianResults, BayesianContext

class NumPyroResults(BayesianResults):
    def encode_solver_results(self, group: h5py.Group):
        samples = self.solver_results
        group['samples'] = samples
        
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        group['samples']
        
class NumPyroFitter(BayesianFitter):
    def make_numpyro_model(self, ctx: BayesianContext):
        import numpyro
        import numpyro.distributions as dist
        
        # Get the model parameters
        params = ctx.model.named_flat_params()
        ctx.make_log_prior_fn()
        
        param_names = list(params.keys())
        param_priors = [param.distribution for param in params.values()]
        
        # Generate feature function and prepare the obs
        self.logger.info("Compiling model and likelihood function...")
        x0 = ctx.model.flat_param_values()
        feature_fn = ctx.make_feature_function()
        feature_fn = jax.jit(feature_fn)
        _y0 = feature_fn(x0)
        
        # Define the numpyro model
        obs_real, obs_imag = jnp.real(ctx.measured_features), jnp.imag(ctx.measured_features)
        def numpyro_model():
            x = jnp.stack([numpyro.sample(param_name, prior) for param_name, prior in zip(param_names, param_priors)])

            y_pred = feature_fn(x)
            y_real, y_imag = jnp.real(y_pred), jnp.imag(y_pred)
            sigma = numpyro.sample('sigma', self.likelihood_params['sigma'].distribution)

            numpyro.sample('obs_real', dist.Normal(y_real, sigma), obs=obs_real)
            numpyro.sample('obs_imag', dist.Normal(y_imag, sigma), obs=obs_imag)       
            
        return numpyro_model

class NumPyroMCMCFitter(NumPyroFitter):
    """
    NumPyro fitter using numpyro.infer.MCMC.
    """        
    def _run(self, ctx: BayesianContext, *, kernel=None, **kwargs) -> NumPyroResults:
        from numpyro.infer import MCMC, NUTS
        
        if kernel is None:
            kernel = NUTS
        
        # Get the model parameters
        param_names = ctx.model.flat_param_names()
        
        # Define the numpyro model
        numpyro_model = self.make_numpyro_model(ctx)
        
        # Run MCMC
        self.logger.info(f'Fitting for {len(param_names)} model parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        rng = jax.random.PRNGKey(42)
        
        kwargs.setdefault('num_warmup', 500)
        kwargs.setdefault('num_samples', 1000)
        mcmc = MCMC(kernel(numpyro_model), **kwargs)
        mcmc.run(rng)
        
        samples = mcmc.get_samples()

        # Posterior means
        x_mean = jnp.stack([samples[param_name].mean() for param_name in param_names])
        fitted_model = ctx.model.with_params(x_mean)
        
        return NumPyroResults(fitted_model=fitted_model, solver_results=samples)
        
class NumPyroNSFitter(NumPyroFitter):
    def _run(self, ctx: BayesianContext, *, constructor_kwargs=None, terminated_kwargs=None) -> NumPyroResults:
        from numpyro.contrib.nested_sampling import NestedSampler
        
        # Get the model parameters
        params = ctx.model.named_flat_params()
        param_names = list(params.keys())
        
        # Define the numpyro model
        numpyro_model = self.make_numpyro_model(ctx)
        
        # Run MCMC
        self.logger.info(f'Fitting for {len(param_names)} model parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        
        ns = NestedSampler(numpyro_model, constructor_kwargs=constructor_kwargs, termination_kwargs=terminated_kwargs)
        ns.run(jax.random.PRNGKey(42))
        
        samples = ns.get_samples(jax.random.PRNGKey(3), num_samples=1000)

        # Posterior means
        x_mean = jnp.stack([samples[param_name].mean() for param_name in param_names])
        fitted_model = ctx.model.with_params(x_mean)
        
        # Return the results
        return NumPyroResults(fitted_model=fitted_model, solver_results=samples)