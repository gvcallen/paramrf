from typing import Any, Callable, Optional
import abc

from jaxtyping import PyTree, Array, Scalar
import equinox as eqx

class SamplerPayload(eqx.Module):
    """The core mathematical payload of a sampling run."""
    #: Stacked latent/unscaled arrays (the samples)
    samples: PyTree[Array]
    
    #: Stacked log-likelihoods or log-posteriors
    fn_values: Array
    
    #: Statistical weights (mostly for Nested/Importance sampling)
    weights: Array | None = None
    
    #: Stacked auxiliary data
    auxes: PyTree[Array] | None = None


class AbstractJointSampler(eqx.Module):
    """Interface for samplers exploring the joint log-posterior (e.g. MCMC-based NUTS or HMC)."""
    @abc.abstractmethod
    def sample(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        raise NotImplementedError


class AbstractSplitSampler(eqx.Module):
    """Interface for samplers needing separate likelihood and prior densities (e.g., modern Nested Sampling)."""
    @abc.abstractmethod
    def sample(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        logprior_fn: Callable[[PyTree], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        raise NotImplementedError


class AbstractHypercubeSampler(eqx.Module):
    """Interface for samplers operating in a unit hypercube (e.g., classical Nested Sampling)."""
    @abc.abstractmethod
    def sample(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        prior_transform_fn: Callable[[PyTree], PyTree],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        has_aux: bool = False,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        raise NotImplementedError