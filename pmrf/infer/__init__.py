"""
Bayesian inference of RF models.

Provides samplers and routines to compute the posterior joint 
distributions of model parameters.
"""

# General API
from pmrf.infer.base import (
    AbstractJointSampler as AbstractJointSampler,
    AbstractSplitSampler as AbstractSplitSampler,
    AbstractHypercubeSampler as AbstractHypercubeSampler,
    AbstractSampler as AbstractSampler,
    is_sampler,
    is_inferer,
    SampleResult,
)
from pmrf.infer.result import InferResult
from pmrf.infer.sample import sample

# Backends
from pmrf.infer.solvers import polychord, blackjax
from pmrf.infer.solvers.polychord import PolyChord
from pmrf.infer.solvers.blackjax import HMC, NUTS, NSS

__all__ = [
    "AbstractJointSampler",
    "AbstractSplitSampler",
    "AbstractHypercubeSampler",
    "AbstractSampler",
    "is_sampler",
    "is_inferer",
    "InferResult",
    "SampleResult",
    "sample",
    "polychord",
    "blackjax",
    "PolyChord",
    "HMC",
    "NUTS",
    "NSS",
]