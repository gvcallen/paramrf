import collections
from typing import Callable, Any, TypeVar, Generic

import jax.numpy as jnp
import jax
import numpy as np
import equinox as eqx

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import field


ModelT = TypeVar('ModelT', bound=Model)


class InferResult(eqx.Module, Generic[ModelT]):
    """
    The result of an inference run.

    Contains the resultant maximum likelihood/maximum a posterior estimates,
    as well as the samples, function values and weights for nested sampling runs.
    """
    #: The maximum likelihood or maximum a posterior of the RF model.
    best_model: ModelT

    #: The maximum likelihood or maximum a posterior of the log-likelihood model.
    best_loglikelihood: Callable[[ModelT, Frequency], jnp.ndarray]
    
    #: A batched model containing the dynamic half of the sampled RF model.
    #: To get the full model, use `equinox.combine(dynamic, static)` with `static_model`.
    #: Only populated for Bayesian sampling algorithms.
    sampled_model: ModelT = None

    #: The static half of `sampled_model`.
    static_model: ModelT = None
    
    #: A batched model containing the dynamic half othe sampled log-likelihood model.
    #: To get the full model, use `equinox.combine(dynamic, static)` with `static_loglikelihood`.
    #: Only populated for Bayesian sampling algorithms.
    sampled_loglikelihood: Callable | eqx.Module = None

    #: The static half of `sampled_loglikelihood`.
    static_loglikelihood: Callable | eqx.Module = None

    #: The function values related to each sample for Bayesian sampling.
    #: Typically, this contains the log likelihood or log posterior values.
    #: Only populated for Bayesian sampling algorithms.
    fn_values: jnp.ndarray = None
    
    #: The weights related to each sample for Bayesian sampling, if any.
    weights: jnp.ndarray = None

    #: The estimated log evidence, if any.
    logevidence: jnp.ndarray = None
    
    #: The estimated error in the log evidence, if any.
    logevidence_err: jnp.ndarray = None
    
    #: The underlying metrics returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    metrics: Any = field(default=None)
       
    # def _prepare_export_data(self, model_prefix: str, likelihood_prefix: str):
    #     """Helper method to extract, format, and check parameter data for export."""
    #     # 1. Cleanly format prefixes
    #     m_prefix = f"{model_prefix}_" if model_prefix else ""
    #     l_prefix = f"{likelihood_prefix}_" if likelihood_prefix else ""
        
    #     model_param_names = [f"{m_prefix}{name}" for name in self.best_model.flat_param_names()]
        
    #     likelihood_param_names = []
        
    #     # 2. Perform Collision Check
    #     param_names = model_param_names + likelihood_param_names
    #     if len(param_names) != len(set(param_names)):
    #         duplicates = [item for item, count in collections.Counter(param_names).items() if count > 1]
    #         raise ValueError(
    #             f"Parameter name collision detected for: {duplicates}. "
    #             "Please provide a unique `model_prefix` and/or `likelihood_prefix` to resolve this."
    #         )
            
    #     # 3. Flatten and vmap
    #     flatten_fn = lambda m: jax.flatten_util.ravel_pytree(m)[0]
    #     sampled_model_params = jax.vmap(flatten_fn)(self.sampled_model)
    #     sampled_loglikelihood_params = jax.vmap(flatten_fn)(self.sampled_loglikelihood)          
        
    #     # 4. Concatenate and cast to standard numpy
    #     sampled_params = np.asarray(jnp.hstack((sampled_model_params, sampled_loglikelihood_params)))
        
    #     return param_names, sampled_params
    
    # def combined_flat_param_values(self) -> jnp.ndarray:
    #     return self._prepare_export_data(model_prefix='model', likelihood_prefix='likelihood')[1]

    # def to_arviz(self, model_prefix='', likelihood_prefix=''):
    #     """Converts the model to Arviz results.

    #     Parameters
    #     ----------
    #     model_prefix : str, optional
    #         A string prefix for the model parameters, by default ''
    #     likelihood_prefix : str, optional
    #         A string prefix for the likelihood parameters, by default ''
    #     """
    #     import arviz as az
        
    #     # 1. Get standardized names and numpy arrays
    #     param_names, sampled_params = self._prepare_export_data(model_prefix, likelihood_prefix)
        
    #     # 2. Construct the ArviZ posterior dictionary
    #     # ArviZ requires shape (n_chains, n_draws). We expand dimensions to add a dummy chain.
    #     posterior_dict = {}
    #     for i, name in enumerate(param_names):
    #         posterior_dict[name] = np.expand_dims(sampled_params[:, i], axis=0)
            
    #     # 3. Extract sample statistics
    #     sample_stats = {
    #         "loglikelihood": np.expand_dims(np.asarray(self.fn_values), axis=0)
    #     }
    #     if self.weights is not None:
    #         sample_stats["weights"] = np.expand_dims(np.asarray(self.weights), axis=0)
            
    #     # 4. Build and return the InferenceData object
    #     return az.from_dict(
    #         posterior=posterior_dict,
    #         sample_stats=sample_stats
    #     )
        
    # def to_anesthetic(self, model_prefix='', likelihood_prefix='', logL_birth=None):
    #     """Converts the model to Anesthetic samples.

    #     Parameters
    #     ----------
    #     model_prefix : str, optional
    #         A string prefix for the model parameters, by default ''
    #     likelihood_prefix : str, optional
    #         A string prefix for the likelihood parameters, by default ''
    #     """        
    #     import pandas as pd
    #     import anesthetic as an
        
    #     # 1. Get standardized names and numpy arrays
    #     param_names, sampled_params = self._prepare_export_data(model_prefix, likelihood_prefix)
        
    #     # 2. Build the core pandas DataFrame
    #     df = pd.DataFrame(sampled_params, columns=param_names)
        
    #     # 3. Extract sample statistics
    #     logL = np.asarray(self.fn_values)
    #     weights = np.asarray(self.weights) if self.weights is not None else None
        
    #     # 4. Determine which Anesthetic object to build
    #     if logL_birth is not None:
    #         return an.NestedSamples(
    #             data=df, 
    #             logL=logL, 
    #             logL_birth=np.asarray(logL_birth), 
    #             weights=weights
    #         )
    #     else:
    #         return an.Samples(
    #             data=df, 
    #             logL=logL, 
    #             weights=weights
    #         )
        