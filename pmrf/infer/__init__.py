"""
Bayesian inference using BlackJAX or PolyChord.

Provides samplers and routines to compute the posterior joint 
distributions of model parameters.
"""

# from pmrf.infer.sample import sample
from pmrf.infer.base import is_sampler, is_inferer
from pmrf.infer.result import InferResult

from pmrf.infer.backends import (
    polychord,
    blackjax,
)

from pmrf.infer.backends.polychord import PolyChord
from pmrf.infer.backends.blackjax import HMC, NUTS, NSS
from pmrf.infer.base import AbstractSplitSampler, AbstractHypercubeSampler, AbstractJointSampler, AbstractSampler

__all__ = [
    "is_sampler",
    "is_inferer",
    "sample",
    "InferResult",
    "polychord",
    "blackjax",
    "PolyChord",
    "HMC",
    "NUTS",
    "NSS",
    "AbstractSplitSampler",
    "AbstractHypercubeSampler",
    "AbstractJointSampler",
    "AbstractSampler",
]