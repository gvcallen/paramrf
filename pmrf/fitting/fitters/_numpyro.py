import jax
import jax.numpy as jnp
from typing import Any
import jax
import jax.numpy as jnp
import h5py

from pmrf.fitting._bayesian import BayesianFitter, BayesianResults

class NumpyroResults(BayesianResults):
    def encode_solver_results(self, group: h5py.Group):
        samples = self.solver_results
        group['samples'] = samples
        
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        group['samples']

class NumpyroFitter(BayesianFitter):
    def run(self, **kwargs) -> NumpyroResults:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        
        # Get the model parameters
        flat_params = self.initial_model.flat_params()
        param_names = [param.name for param in flat_params]
        param_priors = [param.prior for param in flat_params]
        
        # Generate feature function and prepare the obs
        self.logger.info("Compiling model and likelihood function...")
        recon_fn, x0 = self._make_reconstruct_function(flat=True, return_params=True)
        feature_fn = self._make_feature_function(flat=True)
        feature_fn = jax.jit(feature_fn)
        _y0 = feature_fn(x0)
        obs_real, obs_imag = jnp.real(self.measured_features), jnp.imag(self.measured_features)
        
        # Define the numpyro model
        def numpyro_model():
            x = jnp.stack([numpyro.sample(param_name, prior) for param_name, prior in zip(param_names, param_priors)])

            y_pred = feature_fn(x)
            y_real, y_imag = jnp.real(y_pred), jnp.imag(y_pred)
            sigma = numpyro.sample('sigma', self.likelihood_params['sigma'].prior)

            numpyro.sample('obs_real', dist.Normal(y_real, sigma), obs=obs_real)
            numpyro.sample('obs_imag', dist.Normal(y_imag, sigma), obs=obs_imag)
        
        # Run MCMC
        self.logger.info(f'Fitting for {len(param_names)} model parameter(s)...')
        self.logger.info(f'Parameter names: {param_names}')
        kernel = NUTS(numpyro_model)
        rng = jax.random.PRNGKey(42)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(rng)
        
        samples = mcmc.get_samples()

        # Posterior means
        x_mean = jnp.stack([samples[param_name].mean() for param_name in param_names])
        fitted_model = recon_fn(x_mean)
        
        # Return the results
        return NumpyroResults(
            model=fitted_model,
            initial_model=self.initial_model,
            frequency=self.model_frequency,
            measured=self.measured,
            features=self.feature_list,
            logger=self.logger,
            solver_results=samples,
            solver_args=(),
            # solver_kwargs=kwargs,
        )    