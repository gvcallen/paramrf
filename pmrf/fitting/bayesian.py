from abc import ABC
from functools import partial
import matplotlib.pyplot as plt

import skrf
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pmrf.fitting.base import BaseFitter, FitResults
from pmrf.models.model import Model
from pmrf.parameters import Parameter, Uniform
from pmrf.network_collection import NetworkCollection

DefaultSigmaPrior = partial(Uniform, 0.0, 20e-3)

class BayesianFitter(BaseFitter, ABC):
    """
    A base class for Bayesian inference fitters.

    Provides lazily compiled `log_prior` and `log_likelihood` functions.
    """
    def __init__(
        self, 
        model: Model, 
        *, 
        likelihood_kind: str | None = None, 
        likelihood_params: dict[str, Parameter] = None, 
        feature_sigmas: list[str] | None = None, 
        **kwargs,
    ):
        self.likelihood_kind = likelihood_kind
        self.likelihood_params = likelihood_params
        self.feature_sigmas = feature_sigmas
        
        super().__init__(model, **kwargs)

        self._log_prior_fn = None
        self._log_likelihood_fn = None
        
    def run(
        self,
        measured: str | skrf.Network | NetworkCollection,
        **kwargs,
    ) -> tuple[Model, FitResults]:
        # We lazily update the features and likelihood based on the measured data
        sparam_kind = self.feature_kwargs.setdefault('sparam_kind', 'all')
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        
        is_two_port = all([ntwk.nports == 2 for ntwk in measured]) if isinstance(measured, NetworkCollection) else measured.nports == 2
        is_one_port = all([ntwk.nports == 1 for ntwk in measured]) if isinstance(measured, NetworkCollection) else measured.nports == 1
        
        if is_one_port and sparam_kind == 'all':
            sparam_kind = 'reflection'
        likelihood_kind = self.likelihood_kind if self.likelihood_kind is not None else 'multivariate_gaussian' if sparam_kind == 'all' and not is_one_port else 'gaussian'
        
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

        likelihood_params = self.likelihood_params if self.likelihood_params is not None else default_likelihood_params
        features = self.features if self.features is not None else default_features
        feature_sigmas = self.feature_sigmas if self.feature_sigmas is not None else default_feature_sigmas
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
        
        self.likelihood_kind = likelihood_kind
        self.likelihood_params = likelihood_params
        self.feature_sigmas = feature_sigmas
        self.features = features
        
        super().run(measured, **kwargs)

    @property
    def num_params(self) -> int:
        return self.model.num_flat_params + len(self.likelihood_params)
    
    def cdf(self, theta: jnp.ndarray) -> jnp.ndarray:
        if self._cdf_fn is None:
            model_distribution = self.model.distribution()
            num_model_params = self.model.num_flat_params
            
            @jax.jit
            def cdf_fn(u):
                theta_model = model_distribution.cdf(u[0:num_model_params])
                theta_likelihood = jnp.array([param.distribution.cdf(u[num_model_params:][i]) for i, param in enumerate(self.likelihood_params.values())])
                return jnp.concat((theta_model, theta_likelihood))
            self._cdf_fn = cdf_fn
            
        return self._cdf_fn(jnp.array(theta))

    def icdf(self, u: jnp.ndarray) -> jnp.ndarray:
        if self._icdf_fn is None:
            model_distribution = self.model.distribution()
            num_model_params = self.model.num_flat_params
            
            @jax.jit
            def icdf_fn(u):
                theta_model = model_distribution.icdf(u[0:num_model_params])
                theta_likelihood = jnp.array([param.distribution.icdf(u[num_model_params:][i]) for i, param in enumerate(self.likelihood_params.values())])
                return jnp.concat((theta_model, theta_likelihood))
            self._icdf_fn = icdf_fn
            
        return self._icdf_fn(jnp.array(u))

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Lazily compiles and evaluates the combined model + likelihood log-prior."""
        if self._log_prior_fn is None:
            self.logger.debug('Lazily compiling combined log-prior...')
            num_model_params = self.model.num_flat_params
            
            # Use the BaseRunner's logic for the model part
            # but wrap it into a combined JIT closure for the sampler/fitter
            model_dist = self.model.distribution()
            lik_dists = [p.distribution for p in self.likelihood_params.values()]
            
            @jax.jit
            def combined_prior_fn(p):
                # 1. Model Prior (delegated logic)
                lp_model = jnp.sum(model_dist.log_prob(p[0:num_model_params]))
                
                # 2. Likelihood Parameters Prior
                lp_lik = jnp.array([
                    d.log_prob(p[num_model_params:][i]) 
                    for i, d in enumerate(lik_dists)
                ]).sum()
                
                return lp_model + lp_lik
                
            self._log_prior_fn = combined_prior_fn
            
        return self._log_prior_fn(jnp.array(theta))

    def log_likelihood(self, theta: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """
        Lazily compiles and evaluates the log-likelihood.
        Handles expanding 1D parameters into the 2D format expected by the vmapped feature extractor.
        """
        if self._log_likelihood_fn is None:
            # 2. Compile the specific likelihood loop
            if self.likelihood_kind == 'gaussian':
                @jax.jit
                def compiled_ll_fn(theta_and_sigma, target_feats):
                    theta, sigma = theta_and_sigma[0:-1], theta_and_sigma[-1]
                    # Ensure 2D for vmap, extract the 0th batched result
                    y_pred = jnp.real(self.model_features(theta))
                    return dist.Normal(loc=y_pred, scale=sigma).log_prob(jnp.real(target_feats)).sum()
                    
                self._log_likelihood_fn = compiled_ll_fn
                
            elif self.likelihood_kind == 'multivariate_gaussian':
                param_keys = list(self.likelihood_params.keys())
                sigma_indices = jnp.array([param_keys.index(k) for k in self.feature_sigmas])
                num_sigma = len(self.likelihood_params)
                
                @jax.jit
                def compiled_ll_fn(theta_and_sigmas, target_feats):
                    theta, sigmas = theta_and_sigmas[0:-num_sigma], theta_and_sigmas[-num_sigma:]
                    y_pred = jnp.real(self.model_features(theta))
                    scales = sigmas[sigma_indices]
                    return dist.Normal(loc=y_pred, scale=scales).log_prob(jnp.real(target_feats)).sum()
                    
                self._log_likelihood_fn = compiled_ll_fn

        return self._log_likelihood_fn(jnp.array(theta), jnp.array(target))