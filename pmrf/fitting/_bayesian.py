import numpy as np
import jax
import jax.numpy as jnp
import skrf
import numpyro.distributions as dist
import dataclasses

from pmrf.parameters import Parameter, Uniform
from pmrf._model import Model
from pmrf._constants import FeatureInputT
from pmrf.fitting._base import BaseFitter, FitResults


class BayesianResults(FitResults):
    pass
    
class BayesianFitter(BaseFitter):
    """
    **Overview**

    A base class for Bayesian fitting methods.

    This class extends `BaseFitter` by adding the concept of a likelihood function,
    as well as providing support for prior sampling.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | dict[str, skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureInputT | None = None,
        likelihood: str | None = "gaussian",
        likelihood_params: dict[str, Parameter] = None,
        *args, **kwargs
    ) -> None:
        """Initializes the BayesianFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):
                The measured network data to fit the model against.
            frequency (skrf.Frequency | None, optional):
                The frequency axis to perform the fit on. Defaults to `None`.
            features (FeatureT | FeatureListT | None = None, optional):
                The features to extract for comparison.
                Note that note all features are compatibile with all likelihoods,
                but no error checking is currently done for this.
                Defaults to `None`.
        """
        if likelihood != "gaussian" or (likelihood_params is not None and len(likelihood_params) > 1):
            raise Exception("Currently only a gaussian likelihood with a single sigma parameter is supported")
        
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        
        self.likelihood_params = likelihood_params if likelihood_params is not None else {'sigma': Uniform(0.0, 50.0e-3)}
        
    def _flat_params(self) -> jnp.ndarray:        
        sigma_param = dataclasses.replace(self.likelihood_params['sigma'], name='sigma')
        return self.initial_model.flat_params() + [sigma_param]
    
    def _make_prior_transform_function(self, flat=False, numpy_input=False):
        flat = flat or numpy_input
        
        priors = [param.prior for param in self._flat_params()]
        if any(x is None for x in priors):
            raise Exception("Found free parameter without a prior")
        
        prior_fn_jax = lambda hypercube: jnp.array([prior.icdf(hypercube[i]) for i, prior in enumerate(priors)])
        if numpy_input:
            prior_fn = lambda hypercube: np.array(prior_fn_jax(hypercube))
        else:
            prior_fn = prior_fn_jax
        return prior_fn
    
    def _make_logprior_function(self, flat=False, numpy_input=False):
        flat = flat or numpy_input
        
        priors = [param.prior for param in self._flat_params()]
        
        @jax.jit
        def logprior_fn_jax(params: jax.Array) -> float:
            logP = jnp.sum(jnp.array([prior.log_prob(val) for prior, val in zip(priors, params)]))
            return logP            
        
        if numpy_input:
            logprior_fn = lambda x: float(logprior_fn_jax(jnp.array(x)))
        else:
            logprior_fn = logprior_fn_jax
        return logprior_fn
        
    def _make_loglikelihood_function(self, flat=False, numpy_input=False):
        # Old code from polychord fitter
        # feature_fn, x0, recon_fn = make_feature_function(self.initial_model, self.feature_list, self.model_frequency, flat=True, return_params=True, return_recon_fn=True)
        # def jax_likelihood(flat_params_with_sigma) -> jnp.ndarray:
        #     sigma = flat_params_with_sigma[-1]
        #     model_features = feature_fn(flat_params_with_sigma[0:-1])
        #     return gaussian_log_likelihood(self.measured_features, model_features, sigma)        
        # def norm_logpdf(x, loc=0.0, scale=1.0):
        #     return -0.5 * jnp.log(2 * jnp.pi * scale**2) - 0.5 * ((x - loc)**2) / (scale**2)
        # def gaussian_log_likelihood(y_meas, y_model, sigma):
        #     return jnp.sum(norm_logpdf(jnp.real(y_meas), jnp.real(y_model), sigma))
        flat = flat or numpy_input
        
        if not flat:
            raise Exception('Only flat = True currently supported for _make_loglikelihood_function')
        
        feature_fn_jax, x0_jax = self._make_feature_function(flat=flat, return_params=True)
        x0_jax = jnp.array(list(x0_jax) + [self.likelihood_params['sigma'].prior.mean])
        
        obs_real = jnp.real(self.measured_features)
        obs_imag = jnp.imag(self.measured_features)
        
        @jax.jit
        def loglikelihood_fn_jax(params: jnp.ndarray) -> float:
            theta, sigma = params[:-1], params[-1]
            y_pred = feature_fn_jax(theta)
            y_real = jnp.real(y_pred)
            y_imag = jnp.imag(y_pred)

            # Using numpyro for a clean Normal distribution definition
            logL = dist.Normal(loc=y_real, scale=sigma).log_prob(obs_real).sum()
            logL += dist.Normal(loc=y_imag, scale=sigma).log_prob(obs_imag).sum()
            return logL
        
        if numpy_input:
            loglikelihood_fn = lambda x: float(loglikelihood_fn_jax(jnp.array(x)))
            x0 = np.array(x0_jax)
        else:
            loglikelihood_fn = loglikelihood_fn_jax
            x0 = x0_jax
            
        self.logger.info(f"Compiling likelihood function...")
        _ll0 = loglikelihood_fn(x0)
        
        return loglikelihood_fn