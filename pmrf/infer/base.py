"""
Base inference functions and classes.
"""

from typing import Callable, Any, TypeVar, Optional
import collections
import abc


import numpy as np
import jax
from jaxtyping import Array, PyTree, Scalar
import jax.numpy as jnp
import equinox as eqx
import parax as prx

from pmrf.core import Model, Frequency

D = TypeVar('D')

class InferResult(prx.Module):
    """
    The result of an inference run.

    Contains the resultant maximum likelihood estimates, as well as the samples as batched models
    and weights for nested sampling runs.

    Note that all batched objects only contain the relevant free parameters.
    To retrieve a full model, combine it using `eqx.combine` with its dynamic part.
    For example, to retrieve `sampled_models_full`, call `eqx.combine(result.sampled_models, result.model)`.
    """
    #: The RF model containing the maximum likelihood parameters and the posterior over parameters.
    #: Contains the optimized variational posterior distributions for variational inference.
    model: Model

    #: The log likelihood function/model used to calculate the log likelihood during inference.
    #: If the log likelihood was a module with parameters, then this contains
    #: the maximum likelihood log likelihood model.
    loglikelihood: Callable[[Model, Frequency], jnp.ndarray]
    
    #: A batched model containing the sampled models.
    #: Only populated for Bayesian sampling algorithms.
    sampled_models: Model | None = None
    
    #: A batched model containing the sampled log likelihoods if an evaluator was used.
    #: Only populated for Bayesian sampling algorithms.
    sampled_loglikelihoods: Array | None = None
        
    #: The log-likelihood values related to each sample for Bayesian sampling.
    #: Only populated for Bayesian sampling algorithms.
    loglikelihood_values: jnp.ndarray | None = None
    
    #: The weights related to each sample for Bayesian sampling, if any.
    weights: jnp.ndarray | None = None

    #: The estimated log evidence, if any.
    logevidence: Scalar | None = None
    
    #: The estimated error in the log evidence, if any.
    logevidence_err: Scalar | None = None
    
    #: The underlying results object returned by the solver, if any.
    #: May be a stripped-down version of the original results object.
    #: Note saved to file.
    solver_results: Any = prx.constrained(default=None, save=False)
       
    def _prepare_export_data(self, model_prefix: str, likelihood_prefix: str):
        """Helper method to extract, format, and check parameter data for export."""
        # 1. Cleanly format prefixes
        m_prefix = f"{model_prefix}_" if model_prefix else ""
        l_prefix = f"{likelihood_prefix}_" if likelihood_prefix else ""
        
        model_param_names = [f"{m_prefix}{name}" for name in self.model.flat_param_names()]
        
        if isinstance(self.loglikelihood, prx.Module):
            likelihood_param_names = [f"{l_prefix}{name}" for name in self.loglikelihood.flat_param_names()]
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
            
        # 3. Flatten and vmap
        flatten_fn = lambda m: jax.flatten_util.ravel_pytree(m)[0]
        sampled_model_params = jax.vmap(flatten_fn)(self.sampled_models)
        sampled_loglikelihood_params = jax.vmap(flatten_fn)(self.sampled_loglikelihoods)          
        
        # 4. Concatenate and cast to standard numpy
        sampled_params = np.asarray(jnp.hstack((sampled_model_params, sampled_loglikelihood_params)))
        
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
            "loglikelihood": np.expand_dims(np.asarray(self.loglikelihood_values), axis=0)
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
        import pandas as pd
        import anesthetic as an
        
        # 1. Get standardized names and numpy arrays
        param_names, sampled_params = self._prepare_export_data(model_prefix, likelihood_prefix)
        
        # 2. Build the core pandas DataFrame
        df = pd.DataFrame(sampled_params, columns=param_names)
        
        # 3. Extract sample statistics
        logL = np.asarray(self.loglikelihood_values)
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
        

class SamplingResult(eqx.Module):
    """
    A standardized result structure for MCMC and Nested sampling algorithms.
    """
    samples: PyTree[Array]
    #: The stacked trajectory of posterior samples (dead points).
    
    loglikelihoods: Array
    #: A 1D array of log-likelihood values corresponding to each sample.
    
    weights: Array | None = None
    #: The statistical weights associated with each sample.
    #: Only provided by algorithms that require it (e.g. nested sampling).
    
    logevidence: Scalar | None = None
    #: The final estimate of the log-evidence (log Z) of the model.
    #: Only provided by algorithms that support it (e.g. nested sampling).
    
    logevidence_err: Scalar | None = None
    #: The estimated statistical error on the log-evidence.
    #: Only provided by algorithms that support it (e.g. nested sampling).
    
    final_state: Any | None = None
    #: The final internal state of the algorithm.
    
    aux: PyTree[Array] | None = None
    #: Stacked auxiliary data generated during the run.
    
    stats: dict[str, Any] | None = eqx.field(default_factory=dict, static=True)
    #: Static or summary statistics about the run (e.g., number of likelihood evaluations).
    

class AbstractCallableSampler(eqx.Module):
    """
    An interface for JAX-wrapped MCMC and Nested sampling algorithms that require a single `__call__`.
    """
    #: Signifies whether the sampler operates in the unit hypercube.
    #: If True, `prior_fn` must be a transform from a unit hypercube PyTree to a physical space PyTree.
    #: If False, `prior_fn` must return the log-prior probability directly as a scalar.
    requires_hypercube: eqx.AbstractClassVar[bool]

    @abc.abstractmethod
    def __call__(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Scalar],
        prior_fn: Callable[[PyTree, Any], PyTree] | Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        init_samples: Optional[PyTree],
        key: Array,
        args: PyTree[Any],
        options: dict[str, Any],
        max_steps: int | None,
    ) -> SamplingResult:
        """
        Execute the Nested sampling algorithm.

        Parameters
        ----------
        loglikelihood_fn : callable
            A function taking `(params, args)` and returning the scalar log-likelihood.
        prior_fn : callable
            Depending on `requires_hypercube`, either a prior transform function mapping 
            the unit hypercube to physical space, or a function returning the log-prior scalar.
        y0 : PyTree or None
            A prototype PyTree to infer parameter shapes.
        init_samples : PyTree or None
            A batch of PyTrees of the same non-batched shape as `y0` representing initial live points.
            Required for non-hypercube nested samplers.
            For hypercube samplers, these samples should be in the hypercube.
        key : Array
            A JAX PRNGKey for stochastic point generation.
        args : PyTree
            Additional static arguments passed to the likelihood and prior functions.
        options : dict
            Runtime configuration for the sampler.
        max_steps: int | None
            The maximum number of steps the sampler can take, or None for no limit.
        
        Returns
        -------
        SamplingResult
            The structured results containing the samples, log likelihood values, and potentially weights/evidence estimates.
        """
        raise NotImplementedError
    

def is_sampler(x):
    """
    Returns if a solver is suitable for Bayesian sampling in :mod:`pmrf.infer.sample`.

    Returns `True` for :class:`pmrf.infer.AbstractSampler`.
    """    
    return isinstance(x, AbstractCallableSampler)
    

def is_inferer(x):
    """
    Returns if a solver is suitable for Bayesian inference in :mod:`pmrf.infer`.

    Returns `True` for :class:`pmrf.infer.AbstractSampler`.
    """    
    return is_sampler(x)
    

