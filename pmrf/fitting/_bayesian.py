from abc import abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
import skrf
import numpyro.distributions as dist
import dataclasses

from pmrf.constants import FeatureInputT
from pmrf.models import Model
from pmrf.parameters import Parameter, ParameterGroup, Uniform
from pmrf.distributions import TrainableDistribution, TrainableDistributionT
from pmrf.fitting._base import BaseFitter, FitResults

class BayesianResults(FitResults):
    pass

class BayesianSamplingResults(BayesianResults):
    @abstractmethod
    def prior_samples(self) -> jnp.ndarray:
        pass

    @abstractmethod
    def posterior_samples(self) -> jnp.ndarray:
        pass
    
class BayesianFitter(BaseFitter):
    """
    A base class for Bayesian fitting methods.

    This class extends `BaseFitter` by adding the concept of a likelihood function.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | dict[str, skrf.Network],
        *args,
        frequency: skrf.Frequency | None = None,
        features: FeatureInputT | None = None,
        likelihood_kind: str | None = "gaussian",
        likelihood_params: dict[str, Parameter] = None,
        **kwargs
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
                Note that note all features make sense for all likelihoods, but no error checking is done for this.
                Defaults to `None`, in which case real and imaginary feature for all model ports are used.
            likelihood_kind (str, optional):
                The kind of likelihood to use. Defaults to "gaussian" for a one-dimensional Gaussian likelihood
                requiring a single likelihood parameter 'sigma'. Can also be "multivariate_gaussian", in which case either
                multiple standard deviations 'sigma_0', 'sigma_1', ..., 'sigma_N' may be passed, where N is the number of features,
                or an arbitrary number of arbitrarily named likelihood parameters may be passed, along with a list of strings `feature_sigmas`
                of size N containing the names of the likelihood parameters to use for each feature.
            likelihood_params (dict[str, Parameter], optional):
                A dictionary of likelihood parameters to use for the likelihood function.
            train_posteriors (bool, optional):
                Whether or not model posteriors should be trained and set for the fitted model.
        """
        feature_sigmas: list[str] = kwargs.pop('feature_sigmas', None)        

        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        
        if likelihood_kind == 'multivariate_gaussian':
            if feature_sigmas is None:
                raise Exception('feature_sigmas must be passed for multivariate Gaussian likelihoods')
            self.feature_sigmas = feature_sigmas
            self.likelihood_kind = likelihood_kind
            self.likelihood_params = likelihood_params if likelihood_params is not None else {sigma_name: Uniform(0.0, 50.0e-3, name=sigma_name) for sigma_name in feature_sigmas}
        elif likelihood_kind == 'gaussian':        
            if likelihood_params is not None and len(likelihood_params) > 1:
                raise Exception("A gaussian likelihood only has a single likelihood parameter 'sigma'")

            self.likelihood_params = likelihood_params if likelihood_params is not None else {'sigma': Uniform(0.0, 50.0e-3, name='sigma')}
            self.likelihood_kind = likelihood_kind
        else:
            raise Exception(f"Unsupported likelihood kind: {likelihood_kind}")
        
    @property
    def num_params(self) -> int:
        return self.num_model_params + self.num_likelihood_params
    
    @property
    def num_model_params(self) -> int:
        return self.initial_model.num_flat_params
    
    @property
    def num_likelihood_params(self) -> int:
        return len(self.likelihood_params)
    
    def _model_param_names(self) -> list[str]:
        return self.initial_model.flat_param_names()
    
    def _likelihood_param_names(self) -> list[str]:
        return list(self.likelihood_params.keys())
        
    def _flat_param_names(self) -> list[str]:
        return self._model_param_names() + self._likelihood_param_names()
    
    def _make_prior_transform_fn(self, as_numpy=False):
        model_prior = self.initial_model.prior()
        num_model_params = len(self.initial_model.flat_params())
        num_likelihood_params = len(self.likelihood_params)
        
        @jax.jit
        def prior_transform_fn(u):
            theta_model = model_prior.icdf(u[0:num_model_params])
            theta_likelihood = jnp.array([param.distribution.icdf(u[num_model_params:][i]) for i, param in enumerate(self.likelihood_params.values())])
            return jnp.concat((theta_model, theta_likelihood))
            
        if as_numpy:
            prior_transform_fn_jax = prior_transform_fn
            prior_transform_fn = lambda hypercube: np.array(prior_transform_fn_jax(hypercube))
        
        self.logger.info('Compiling prior transform...')
        _prior = prior_transform_fn(jnp.array([0.5] * (num_model_params + num_likelihood_params)))
        
        return prior_transform_fn
    
    def _make_log_prior_fn(self, as_numpy=False):
        model_prior = self.initial_model.prior()
        num_model_params = self.initial_model.num_flat_params
        num_likelihood_params = len(self.likelihood_params)
        
        @jax.jit
        def logprior_fn(params: jax.Array) -> float:
            logprob_model = model_prior.log_prob(params[0:num_model_params])
            logprob_likelihood = jnp.array([param.distribution.log_prob(params[num_model_params:][i]) for i, param in enumerate(self.likelihood_params.values())])
            return jnp.sum(logprob_model) + jnp.sum(logprob_likelihood)
        
        if as_numpy:
            logprior_fn_jax = logprior_fn
            logprior_fn = lambda x: float(logprior_fn_jax(jnp.array(x)))

        self.logger.info('Compiling log prior...')
        _prior = logprior_fn(jnp.array([0.5] * (num_model_params + num_likelihood_params)))
            
        return logprior_fn
        
    def _make_log_likelihood_fn(self, as_numpy=False):
        if self.likelihood_kind == 'gaussian':        
            log_likelihood_fn = self._make_gaussian_log_likelihood_fn()        
        elif self.likelihood_kind == 'multivariate_gaussian':
            log_likelihood_fn = self._make_multivariate_gaussian_log_likelihood_fn()
        else:
            raise Exception(f"Unsupported likelihood kind: {self.likelihood_kind}")

        x0 = jnp.array(list(self.initial_model.flat_params()) + [param.distribution.mean for param in self.likelihood_params.values()])
        if as_numpy:
            log_likelihood_fn_jax = log_likelihood_fn
            log_likelihood_fn = lambda x: float(log_likelihood_fn_jax(jnp.array(x)))
            x0 = np.array(x0)
            
        self.logger.info(f"Compiling likelihood function...")
        _log_likelihood = log_likelihood_fn(x0)

        return log_likelihood_fn
    
    def _make_gaussian_log_likelihood_fn(self):
        feature_fn_jax = self._make_feature_function() 
        
        @jax.jit
        def loglikelihood_fn(flat_params) -> float:
            theta, sigma = flat_params[0:-1], flat_params[-1]
            y_pred = jnp.real(feature_fn_jax(theta))
            y_meas = jnp.real(self.measured_features)
            logL = dist.Normal(loc=y_pred, scale=sigma).log_prob(y_meas).sum()
            return logL
        
        return loglikelihood_fn
    
    def _make_multivariate_gaussian_log_likelihood_fn(self):
        feature_fn_jax = self._make_feature_function()
        
        @jax.jit
        def loglikelihood_fn(flat_params) -> float:
            num_sigma = len(self.likelihood_params)
            theta, sigmas = flat_params[0:-num_sigma], flat_params[-num_sigma:]
            y_pred = jnp.real(feature_fn_jax(theta))
            y_meas = jnp.real(self.measured_features)
            
            param_keys = list(self.likelihood_params.keys())
            sigma_indices = jnp.array([param_keys.index(key) for key in self.feature_sigmas])
            scales = sigmas[sigma_indices]
            logL = dist.Normal(loc=y_pred, scale=scales).log_prob(y_meas).sum()
            return logL
        
        return loglikelihood_fn
    
    @abstractmethod
    def update_posteriors(self) -> BayesianResults:
        raise Exception('Update posteriors is an abstract method and must be implemented in a Bayesian fitter')
    
class BayesianSamplingFitter(BayesianFitter):
    def run(self, *args, **kwargs) -> BayesianSamplingResults:
        return super().run(*args, **kwargs)

    def update_posteriors(self, dist: TrainableDistributionT, **train_kwargs) -> BayesianSamplingResults:
        results: BayesianSamplingResults = self.results

        posterior_samples: jnp.ndarray = results.posterior_samples()
        model_param_names: list[str] = self._model_param_names()
        model_posterior_samples: jnp.ndarray = posterior_samples[:,0:self.num_model_params]
        model_posterior_dist = dist.from_samples(model_posterior_samples)
        model_param_group = ParameterGroup(model_param_names, model_posterior_dist)
        
        results.fitted_model = results.fitted_model.with_param_groups(model_param_group)

        return results