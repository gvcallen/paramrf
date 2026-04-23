"""
Bayesian inference using Inferix.

Provides samplers and routines to compute the posterior joint 
distributions of model parameters given measured data.
"""

from pmrf.infer.sample import sample
from pmrf.infer.base import is_inferer, InferResult
from pmrf.infer.polychord import PolyChord
from pmrf.constants import Inferer

__all__ = [
    "is_inferer",
    "sample",
    "InferResult",
    "Inferer",
    "PolyChord",
]