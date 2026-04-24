"""
Bayesian inference using BlackJAX or PolyChord.

Provides samplers and routines to compute the posterior joint 
distributions of model parameters.
"""

from pmrf.infer.sample import sample
from pmrf.infer.base import is_sampler, is_inferer, InferResult
from pmrf.infer.polychord import PolyChord
from pmrf.infer.blackjax import HMC, NUTS, NSS
from pmrf.infer.base import AbstractCallableSampler

__all__ = [
    "is_sampler",
    "is_inferer",
    "sample",
    "InferResult",
    "PolyChord",
    "HMC",
    "NUTS",
    "NSS",
    "AbstractCallableSampler",
]