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

from pmrf.models import Model
from pmrf.frequency import Frequency

D = TypeVar('D')

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
    
AbstractSampler = AbstractJointSampler | AbstractSplitSampler | AbstractHypercubeSampler


def is_sampler(x):
    """
    Returns if a solver is suitable for Bayesian sampling in :mod:`pmrf.infer.sample`.

    Returns `True` for :class:`pmrf.infer.AbstractSampler`.
    """    
    return isinstance(x, AbstractSampler)
    

def is_inferer(x):
    """
    Returns if a solver is suitable for Bayesian inference in :mod:`pmrf.infer`.

    Returns `True` for :class:`pmrf.infer.AbstractSampler`.
    """    
    return is_sampler(x)
    

