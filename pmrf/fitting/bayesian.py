from abc import ABC
from functools import partial
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from pmrf.fitting.base import BaseFitter
from pmrf.fitting.results import FitResults
from pmrf.models.model import Model
from pmrf.parameters import Parameter, Uniform

DefaultSigmaPrior = partial(Uniform, 0.0, 20e-3)

class BayesianFitter(BaseFitter, ABC):
    """
    A base class for Bayesian inference fitters.

    Provides lazily compiled `logprior` and `loglikelihood` functions.
    """
    def __init__(
        self, 
        model: Model, 
        *, 
        likelihood_kind: str | None = None, 
        likelihood_params: dict[str, Parameter] = None, 
        feature_sigmas: list[str] | None = None, 
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        # Model introspection for defaults
        num_ports = max([max(m, n) for m, n in model.port_tuples]) + 1 if model.port_tuples else 1
        is_one_port = num_ports == 1
        is_two_port = num_ports == 2

        if is_one_port and self.sparam_kind == 'all':
            self.sparam_kind = 'reflection'
            
        self.likelihood_kind = likelihood_kind or ('multivariate_gaussian' if self.sparam_kind == 'all' and not is_one_port else 'gaussian')

        # Feature and Parameter Defaults Mapping
        if self.likelihood_kind == 'gaussian':
            self.likelihood_params = likelihood_params or {'sigma': DefaultSigmaPrior(name='sigma')}
            if len(self.likelihood_params) > 1:
                raise ValueError("A gaussian likelihood only takes a single parameter 'sigma'")
                
            if self.features is None:
                if is_two_port:
                    if self.sparam_kind == 'all':
                        self.features = ['s11_re', 's11_im', 's12_re', 's12_im', 's21_re', 's21_im', 's22_re', 's22_im']
                    elif self.sparam_kind == 'reflection':
                        self.features = ['s11_re', 's11_im', 's22_re', 's22_im']
                    elif self.sparam_kind == 'transmission':
                        self.features = ['s12_re', 's12_im', 's21_re', 's21_im']
                else:
                    self.features = ['s_re', 's_im']
                    
        elif self.likelihood_kind == 'multivariate_gaussian':
            if is_two_port and self.sparam_kind == 'all':
                self.features = self.features or ['s11_re', 's11_im', 's12_re', 's12_im', 's21_re', 's21_im', 's22_re', 's22_im']
                self.feature_sigmas = feature_sigmas or ['sigma_gamma', 'sigma_gamma', 'sigma_tau', 'sigma_tau', 'sigma_tau', 'sigma_tau', 'sigma_gamma', 'sigma_gamma']
                self.likelihood_params = likelihood_params or {s: DefaultSigmaPrior(name=s) for s in set(self.feature_sigmas)}
            elif is_two_port:
                raise ValueError("No need to use multivariate gaussian when fitting only transmission or reflection")
            else:
                self.feature_sigmas = feature_sigmas
                
            if self.feature_sigmas is None:
                raise ValueError("feature_sigmas must be provided for multivariate Gaussian likelihoods")
        else:
            raise ValueError(f"Unsupported likelihood kind: {self.likelihood_kind}")

        # Lazy Compilation Caches
        self._log_prior_fn = None
        self._log_likelihood_fn = None

    @property
    def num_params(self) -> int:
        return self.model.num_flat_params + len(self.likelihood_params)

    def log_prior(self, params: jnp.ndarray) -> jnp.ndarray:
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
            
        return self._log_prior_fn(jnp.array(params))

    def log_likelihood(self, flat_params: jnp.ndarray, target_features: jnp.ndarray) -> jnp.ndarray:
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

        return self._log_likelihood_fn(jnp.array(flat_params), jnp.array(target_features))