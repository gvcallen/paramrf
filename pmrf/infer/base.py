"""
Base inference functions and classes.
"""

from typing import Callable

import jax
from jaxtyping import Array
import jax.numpy as jnp
import parax as prx
from parax import partition
import inferix as infx

from pmrf.core import Model, Frequency

class InferResult(prx.Module):
    """
    The result of an inference run.
    """
    #: The RF model containing the maximum likelihood parameters and the posterior over parameters.
    model: Model

    #: The log likelihood function/model used to calculate the log likelihood during sampling.
    #: If the log likelihood was a module with parameters, then this contains
    #: the maximum likelihood log likelihood model.
    log_likelihood: Callable[[Model, Frequency], jnp.ndarray]
    
    #: A batched model containing the sampled models.
    sampled_models: Model
    
    #: A batched model containing the sampled log likelihoods if an evaluator was used.
    sampled_log_likelihoods: Array
        
    #: The log-likelihood values related to each sample.
    log_likelihood_values: jnp.ndarray
    
    #: The weights related to each sample, if any.
    weights: jnp.ndarray | None = None
    
    #: The underlying results object returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    solver_results: infx.Result = None
       
    def _prepare_export_data(self, model_prefix: str, likelihood_prefix: str):
        """Helper method to extract, format, and check parameter data for export."""
        import collections
        import numpy as np
        
        # 1. Cleanly format prefixes
        m_prefix = f"{model_prefix}_" if model_prefix else ""
        l_prefix = f"{likelihood_prefix}_" if likelihood_prefix else ""
        
        model_param_names = [f"{m_prefix}{name}" for name in self.model.flat_param_names()]
        
        if isinstance(self.log_likelihood, prx.Module):
            likelihood_param_names = [f"{l_prefix}{name}" for name in self.log_likelihood.flat_param_names()]
        else:
            likelihood_param_names = []
        
        # 2. Perform Collision Check
        param_names = model_param_names + likelihood_param_names
        if len(param_names) != len(set(param_names)):
            duplicates = [item for item, count in collections.Counter(param_names).items() if count > 1]
            raise ValueError(
                f"Parameter name collision detected for: {duplicates}. "
                "Please provide a unique `model_prefix` and/or `likelihood_prefix` to resolve this."
            )
            
        # 3. Partition out static variables to isolate batched dynamic parameters
        dynamic_models, _ = partition(self.sampled_models)
        dynamic_likelihoods, _ = partition(self.sampled_log_likelihoods)

        # 4. Flatten and vmap
        flatten_fn = lambda m: jax.flatten_util.ravel_pytree(m)[0]
        sampled_model_params = jax.vmap(flatten_fn)(dynamic_models)
        sampled_log_likelihood_params = jax.vmap(flatten_fn)(dynamic_likelihoods)          
        
        # 5. Concatenate and cast to standard numpy
        sampled_params = np.asarray(jnp.hstack((sampled_model_params, sampled_log_likelihood_params)))
        
        return param_names, sampled_params
    
    def combined_flat_param_values(self) -> jnp.ndarray:
        return self._prepare_export_data(model_prefix='model', likelihood_prefix='likelihood')[1]

    def to_arviz(self, model_prefix='', likelihood_prefix=''):
        """Converts the model to Arviz results.

        Parameters
        ----------
        model_prefix : str, optional
            A string prefix for the model parameters, by default ''
        likelihood_prefix : str, optional
            A string prefix for the likelihood parameters, by default ''
        """
        import numpy as np
        import arviz as az
        
        # 1. Get standardized names and numpy arrays
        param_names, sampled_params = self._prepare_export_data(model_prefix, likelihood_prefix)
        
        # 2. Construct the ArviZ posterior dictionary
        # ArviZ requires shape (n_chains, n_draws). We expand dimensions to add a dummy chain.
        posterior_dict = {}
        for i, name in enumerate(param_names):
            posterior_dict[name] = np.expand_dims(sampled_params[:, i], axis=0)
            
        # 3. Extract sample statistics
        sample_stats = {
            "log_likelihood": np.expand_dims(np.asarray(self.log_likelihood_values), axis=0)
        }
        if self.weights is not None:
            sample_stats["weights"] = np.expand_dims(np.asarray(self.weights), axis=0)
            
        # 4. Build and return the InferenceData object
        return az.from_dict(
            posterior=posterior_dict,
            sample_stats=sample_stats
        )
        
    def to_anesthetic(self, model_prefix='', likelihood_prefix='', logL_birth=None):
        """Converts the model to Anesthetic samples.

        Parameters
        ----------
        model_prefix : str, optional
            A string prefix for the model parameters, by default ''
        likelihood_prefix : str, optional
            A string prefix for the likelihood parameters, by default ''
        """        
        import numpy as np
        import pandas as pd
        import anesthetic as an
        
        # 1. Get standardized names and numpy arrays
        param_names, sampled_params = self._prepare_export_data(model_prefix, likelihood_prefix)
        
        # 2. Build the core pandas DataFrame
        df = pd.DataFrame(sampled_params, columns=param_names)
        
        # 3. Extract sample statistics
        logL = np.asarray(self.log_likelihood_values)
        weights = np.asarray(self.weights) if self.weights is not None else None
        
        # 4. Determine which Anesthetic object to build
        if logL_birth is not None:
            return an.NestedSamples(
                data=df, 
                logL=logL, 
                logL_birth=np.asarray(logL_birth), 
                weights=weights
            )
        else:
            return an.Samples(
                data=df, 
                logL=logL, 
                weights=weights
            )
            
def is_inferer(x):
    """
    Returns if a solver is suitable for Bayesian inference in :mod:`pmrf.infer`.

    Returns `True` for :class:`pmrf.infer.PolyChord` and :class:`inferix.AbstractSampler`.
    """    
    from pmrf.infer.polychord import PolyChord
    return isinstance(x, PolyChord | infx.AbstractSampler)