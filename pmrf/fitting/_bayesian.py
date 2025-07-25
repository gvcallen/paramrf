import numpy as np
import jax
import jax.numpy as jnp
import skrf
import numpyro.distributions as dist
import dataclasses

from pmrf.parameters import Parameter, ParameterGroup, Uniform
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
        self.likelihood_params = likelihood_params if likelihood_params is not None else {'sigma': Uniform(0.0, 50.0e-3, name='sigma')}
        
    def _params(self) -> dict[str, Parameter]:
        return self.initial_model.params(flat=True) | self.likelihood_params
    
    def _make_inverse_cdf_function(self, flat=False, numpy_input=False):
        flat = flat or numpy_input
        
        param_groups: list[ParameterGroup] = self.initial_model.param_groups(flat=True)
        param_names = list(self._params().keys())
        
        # The first case is for independent priors (each group maps to one parameter) whereas the second case is for correlated priors
        if len(param_groups) == len(param_names):
            priors = [param.prior for param in self._params()]
            if any(x is None for x in priors):
                raise Exception("Found free parameter without a prior")
            prior_fn_jax = lambda hypercube: jnp.array([prior.icdf(hypercube[i]) for i, prior in enumerate(priors)])
            prior_fn_jax = jax.jit(prior_fn_jax)
        else:
            @jax.jit
            def prior_fn_jax(u: jnp.ndarray):
                # We assign groups of d hypercube values to corresponding groups of physical values
                name_to_hypercube_value = {name: u[i] for i, name in enumerate(param_names)}
                name_to_physical_value = {name: None for name in param_names}
                
                # First, we initialize the likelihood parameters (taken from the end of the hypercube)
                for i, (likelihood_param_name, likelihood_param_value) in enumerate(self.likelihood_params.items()):
                    name_to_physical_value[likelihood_param_name] = likelihood_param_value.prior.icdf(u[-i])
                
                # Then we run through the parameter groups, collect the d hypercube parameters into an array g per group,
                # and use the icdf of the group prior to get the physical parameters for that group
                for param_group in param_groups:
                    group_param_names = list(param_group.params.keys())
                    g = [name_to_hypercube_value[name] for name in group_param_names if name in param_names]
                    
                    # Either all parameters or no parameters must be present - the inverse transform is not partially defined
                    if len(g) == 0:
                        continue
                    elif len(g) != len(group_param_names):
                        raise Exception('Cannot use correlated priors where some parameters are fixed')
                    
                    g = jnp.array(g)
                    param_values = param_group.prior.icdf(g)
                    for i, name in enumerate(group_param_names):
                        name_to_physical_value[name] = param_values[i]
                        
                # Should probably check this outside of the function
                if any(value is None for value in name_to_physical_value.values()):
                    raise Exception('Parameter found that did not belong to a parameter groups')
                
                # Return the physical values
                return jnp.array(list(name_to_physical_value.values()))
        
        _prior_vals = prior_fn_jax(jnp.array([0.5] * len(param_names)))
        if numpy_input:
            prior_fn = lambda hypercube: np.array(prior_fn_jax(hypercube))
        else:
            prior_fn = prior_fn_jax
        return prior_fn
    
    def _make_log_prior_function(self, flat=False, numpy_input=False):
        # TODO cater for parameter groups
        flat = flat or numpy_input
        
        priors = [param.prior for param in self._params()]
        
        @jax.jit
        def logprior_fn_jax(params: jax.Array) -> float:
            logP = jnp.sum(jnp.array([prior.log_prob(val) for prior, val in zip(priors, params)]))
            return logP            
        
        if numpy_input:
            logprior_fn = lambda x: float(logprior_fn_jax(jnp.array(x)))
        else:
            logprior_fn = logprior_fn_jax
        return logprior_fn
        
    def _make_log_likelihood_function(self, flat=False, numpy_input=False):
        flat = flat or numpy_input
        
        if not flat:
            raise Exception('Only flat = True currently supported for _make_loglikelihood_function')
        
        feature_fn_jax, x0_jax = self._make_feature_function(flat=flat, return_params=True)
        x0_jax = jnp.array(list(x0_jax) + [self.likelihood_params['sigma'].prior.mean])
        
        def norm_logpdf(x, loc=0.0, scale=1.0):
            return -0.5 * jnp.log(2 * jnp.pi * scale**2) - 0.5 * ((x - loc)**2) / (scale**2)
        def gaussian_log_likelihood(y_meas, y_model, sigma):
            return jnp.sum(norm_logpdf(jnp.real(y_meas), jnp.real(y_model), sigma))        
        def loglikelihood_fn_jax(flat_params_with_sigma) -> jnp.ndarray:
            sigma = flat_params_with_sigma[-1]
            model_features = feature_fn_jax(flat_params_with_sigma[0:-1])
            return gaussian_log_likelihood(self.measured_features, model_features, sigma)        
        
        # obs_real = jnp.real(self.measured_features)
        # obs_imag = jnp.imag(self.measured_features)
        # @jax.jit
        # def loglikelihood_fn_jax(params: jnp.ndarray) -> float:
        #     theta, sigma = params[:-1], params[-1]
        #     y_pred = feature_fn_jax(theta)
        #     y_real = jnp.real(y_pred)
        #     # y_imag = jnp.imag(y_pred)

        #     # Using numpyro for a clean Normal distribution definition
        #     logL = dist.Normal(loc=y_real, scale=sigma).log_prob(obs_real).sum()
        #     # logL += dist.Normal(loc=y_imag, scale=sigma).log_prob(obs_imag).sum()
        #     return logL
        
        
        if numpy_input:
            loglikelihood_fn = lambda x: float(loglikelihood_fn_jax(jnp.array(x)))
            x0 = np.array(x0_jax)
        else:
            loglikelihood_fn = loglikelihood_fn_jax
            x0 = x0_jax
            
        self.logger.info(f"Compiling likelihood function...")
        _ll0 = loglikelihood_fn(x0)
        
        return loglikelihood_fn