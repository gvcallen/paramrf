import dataclasses
from dataclasses import dataclass
from functools import partial
from abc import abstractmethod

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
import skrf
import numpyro.distributions as dist
import matplotlib.pyplot as plt

from pmrf.network_collection import NetworkCollection
from pmrf._util import RANK, wait_for_all_ranks
from pmrf.models import Model
from pmrf.parameters import Parameter, ParameterGroup, Uniform
from pmrf.distributions.trainable import TrainableDistributionT
from pmrf.fitting.base import BaseFitter, FitResults, FitContext

DefaultSigmaPrior = partial(Uniform, 0.0, 20e-3)

class BayesianResults(FitResults):
    @abstractmethod
    def prior_samples(self, equal_weights=False) -> jnp.ndarray:
        pass

    @abstractmethod
    def posterior_samples(self, equal_weights=False) -> jnp.ndarray:
        pass

    @abstractmethod
    def weights(self) -> jnp.ndarray:
        pass
    
    def fit_posterior(self, train_dist: TrainableDistributionT | None = None, equal_weights=False, drift_sigma=0.0, boost_method=None, boost_samples=10000, **train_kwargs):
        param_names: list[str] = self.fitted_model.flat_param_names()
        training_data: jnp.ndarray = self.posterior_samples(equal_weights=equal_weights)[:,0:self.fitted_model.num_flat_params]        

        if drift_sigma != 0.0:
            if boost_method == 'kde':
                from margarine.kde import KDE
                kde = KDE(training_data)
                kde.generate_kde()
                training_data = kde.sample(boost_samples)
            elif boost_method != None:
                raise Exception('Unknown posterior training data boost method')
                
            scale = np.abs(np.mean(training_data, axis=0)) * drift_sigma
            training_data += np.random.normal(loc=0.0, scale=scale, size=training_data.shape)

        if train_dist is None:
            from pmrf.distributions import MargarineMAFDistribution
            train_dist = MargarineMAFDistribution
        
        if equal_weights:
            dist = train_dist.from_samples(training_data, **train_kwargs)
        else:
            weights = self.weights()
            dist = train_dist.from_weighted_samples(training_data, weights, **train_kwargs)
        param_group = ParameterGroup(param_names, dist)
        
        self.fitted_model = self.fitted_model.with_param_groups(param_group)
        
@dataclass
class BayesianContext(FitContext):
    likelihood_kind: str = None
    likelihood_params: dict[str, Parameter] = None
    feature_sigmas: list[str] | None = None
    
    @property
    def num_params(self) -> int:
        return self.num_model_params + self.num_likelihood_params
    
    @property
    def num_model_params(self) -> int:
        return self.model.num_flat_params
    
    @property
    def num_likelihood_params(self) -> int:
        return len(self.likelihood_params)
    
    def likelihood_param_names(self) -> list[str]:
        return list(self.likelihood_params.keys())
        
    def flat_param_names(self) -> list[str]:
        return self.model_param_names() + self.likelihood_param_names()
    
    def make_prior_transform_fn(self, as_numpy=False):
        model_prior = self.model.distribution()
        num_model_params = len(self.model.flat_params())
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
    
    def make_log_prior_fn(self, as_numpy=False):
        model_prior = self.model.distribution()
        num_model_params = self.model.num_flat_params
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
        
    def make_log_likelihood_fn(self, as_numpy=False):
        if self.likelihood_kind == 'gaussian':        
            log_likelihood_fn = self.make_gaussian_log_likelihood_fn()        
        elif self.likelihood_kind == 'multivariate_gaussian':
            log_likelihood_fn = self.make_multivariate_gaussian_log_likelihood_fn()
        else:
            raise Exception(f"Unsupported likelihood kind: {self.likelihood_kind}")

        x0 = jnp.array(list(self.model.flat_params()) + [param.distribution.mean for param in self.likelihood_params.values()])
        if as_numpy:
            log_likelihood_fn_jax = log_likelihood_fn
            log_likelihood_fn = lambda x: float(log_likelihood_fn_jax(jnp.array(x)))
            x0 = np.array(x0)
            
        self.logger.info(f"Compiling likelihood function...")
        _log_likelihood = log_likelihood_fn(x0)

        return log_likelihood_fn
    
    def make_gaussian_log_likelihood_fn(self):
        feature_fn_jax = self.make_feature_function() 
        
        @jax.jit
        def loglikelihood_fn(flat_params) -> float:
            theta, sigma = flat_params[0:-1], flat_params[-1]
            y_pred = jnp.real(feature_fn_jax(theta))
            y_meas = jnp.real(self._active_measured_features)
            logL = dist.Normal(loc=y_pred, scale=sigma).log_prob(y_meas).sum()
            return logL
        
        return loglikelihood_fn
    
    def make_multivariate_gaussian_log_likelihood_fn(self):
        feature_fn_jax = self.make_feature_function()
        
        @jax.jit
        def loglikelihood_fn(flat_params) -> float:
            num_sigma = len(self.likelihood_params)
            theta, sigmas = flat_params[0:-num_sigma], flat_params[-num_sigma:]
            y_pred = jnp.real(feature_fn_jax(theta))
            y_meas = jnp.real(self._active_measured_features)
            
            param_keys = list(self.likelihood_params.keys())
            sigma_indices = jnp.array([param_keys.index(key) for key in self.feature_sigmas])
            scales = sigmas[sigma_indices]
            logL = dist.Normal(loc=y_pred, scale=scales).log_prob(y_meas).sum()
            return logL
        
        return loglikelihood_fn    
    
    
class BayesianFitter(BaseFitter):
    """
    A base class for Bayesian fitting methods.

    This class extends `BaseFitter` by adding the concept of a likelihood function.
    """
    def __init__(
        self,
        model: Model,
        *,
        likelihood_kind: str | None = None,
        likelihood_params: dict[str, Parameter] = None,
        feature_sigmas: list[str] | None = None,
        **kwargs
    ) -> None:
        """Initializes the BayesianFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            likelihood_kind (str, optional):
                The kind of likelihood to use. Can be either 'gaussian' or 'multivariate_gaussian'.
                Defaults internally to 'gaussian' for one-port fits, and 'multivariate_gaussian' for greater port fits.
                For 'gaussian', a single likelihood parameter, 'sigma', is needed. For 'multivariate_gaussian',
                either multiple standard deviations 'sigma_0', 'sigma_1', ..., 'sigma_N' may be passed, where N is the number of features,
                or an arbitrary number of arbitrarily named likelihood parameters may be passed, along with a list of strings `feature_sigmas`
                of size N containing the names of the likelihood parameters to use for each feature.
            likelihood_params (dict[str, Parameter], optional):
                A dictionary of likelihood parameters to use for the likelihood function.
            feature_sigmas (list[str], optional):
                A list of sigma names for each feature. Only used when `likelihood_kind` is 'multivariate_gaussian'.
        """
        self.likelihood_kind = likelihood_kind
        self.likelihood_params = likelihood_params
        self.feature_sigmas = feature_sigmas

        super().__init__(model, **kwargs)    
    
    def create_context(self, measured, *, likelihood_kind=None, likelihood_params=None, feature_sigmas=None, **kwargs) -> BayesianContext:
        features = kwargs.pop('features', None) or self.features
        likelihood_kind = likelihood_kind or self.likelihood_kind
        likelihood_params = likelihood_params or self.likelihood_params
        feature_sigmas = feature_sigmas or self.feature_sigmas
        
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        
        is_two_port = all([ntwk.nports == 2 for ntwk in measured]) if isinstance(measured, NetworkCollection) else measured.nports == 2
        is_one_port = all([ntwk.nports == 1 for ntwk in measured]) if isinstance(measured, NetworkCollection) else measured.nports == 1
        
        if is_one_port and sparam_kind == 'all':
            sparam_kind = 'reflection'
        likelihood_kind = likelihood_kind if likelihood_kind is not None else 'multivariate_gaussian' if sparam_kind == 'all' and not is_one_port else 'gaussian'
        
        default_likelihood_params = default_features = default_feature_sigmas = None
        if likelihood_kind == 'gaussian':
            default_likelihood_params = {'sigma': DefaultSigmaPrior()}
            if is_two_port:
                if sparam_kind == 'all':
                    default_features = ['s11_re', 's11_im', 's12_re', 's12_im', 's21_re', 's21_im', 's22_re', 's22_im']
                elif sparam_kind == 'reflection':
                    default_features = ['s11_re', 's11_im', 's22_re', 's22_im']
                elif sparam_kind == 'transmission':
                    default_features = ['s12_re', 's12_im', 's21_re', 's21_im']
            else:
                default_features = ['s_re', 's_im']
        elif likelihood_kind == 'multivariate_gaussian':
            if is_two_port:
                default_likelihood_params = {sigma_name: DefaultSigmaPrior() for sigma_name in ['sigma_gamma', 'sigma_tau']}
                if sparam_kind == 'all':
                    default_features = ['s11_re', 's11_im', 's12_re', 's12_im', 's21_re', 's21_im', 's22_re', 's22_im']
                    default_feature_sigmas = ['sigma_gamma', 'sigma_gamma', 'sigma_tau', 'sigma_tau', 'sigma_tau', 'sigma_tau', 'sigma_gamma', 'sigma_gamma']
                else:
                    raise Exception('No need to use multivariate gaussian when fitting only transmission or reflection coefficients')
            else:
                pass

        likelihood_params = likelihood_params if likelihood_params is not None else default_likelihood_params
        features = features if features is not None else default_features
        feature_sigmas = feature_sigmas if feature_sigmas is not None else default_feature_sigmas
        
        base_ctx = super().create_context(measured, features=features, **kwargs)
    
        if likelihood_kind == 'multivariate_gaussian':
            if feature_sigmas is None:
                raise Exception('feature_sigmas must be passed for multivariate Gaussian likelihoods')
            likelihood_params = likelihood_params if likelihood_params is not None else {sigma_name: DefaultSigmaPrior(name=sigma_name) for sigma_name in feature_sigmas}
        elif likelihood_kind == 'gaussian':        
            if likelihood_params is not None and len(likelihood_params) > 1:
                raise Exception("A gaussian likelihood only has a single likelihood parameter 'sigma'")

            likelihood_params = likelihood_params if likelihood_params is not None else {'sigma': DefaultSigmaPrior(name='sigma')}
        else:
            raise Exception(f"Unsupported likelihood kind: {likelihood_kind}")
        
        return BayesianContext(
            model=base_ctx.model,
            measured=base_ctx.measured,
            frequency=base_ctx.frequency,
            features=base_ctx.features,
            measured_features=base_ctx.measured_features,
            output_path=base_ctx.output_path,
            output_root=base_ctx.output_root,
            sparam_kind=base_ctx.sparam_kind,
            likelihood_kind=likelihood_kind,
            likelihood_params=likelihood_params,
            feature_sigmas=feature_sigmas,
        )
        
    def run(self, ctx: BayesianContext, plot_params=False, fit_posterior=False, fit_posterior_dist=None, fit_posterior_kwargs=None, *args, **kwargs) -> BayesianResults:
        user_callback = kwargs.get('callback', None)
        fit_posterior_kwargs = fit_posterior_kwargs or {}
        
        def callback(results: BayesianResults):
            nonlocal user_callback
            nonlocal fit_posterior_dist
            
            if RANK == 0:
                from pmrf.distributions import MargarineMAFDistribution
                fit_posterior_dist = fit_posterior_dist or MargarineMAFDistribution
                results.fit_posterior(fit_posterior_dist, **fit_posterior_kwargs)
            wait_for_all_ranks()
            if user_callback:
                kwargs['callback'](results)
        
        if fit_posterior:
            user_callback = kwargs.pop('callback', None)
            kwargs['callback'] = callback

        results: BayesianResults = super().run(*args, **kwargs)

        if plot_params:
            results.plot_params()
            plt.savefig(f'{ctx.output_path}/params.png')

        return results        