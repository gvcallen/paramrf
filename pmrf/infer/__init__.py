"""
Bayesian inference using Inferix.

Provides samplers and routines to compute the posterior joint 
distributions of model parameters given measured data.
"""

from pmrf.infer.sample import sample
from pmrf.infer.condition import condition
from pmrf.infer.result import InferResult
from pmrf.infer.solvers import is_inferer
from pmrf.constants import Inferer

__all__ = [
    "is_inferer",
    "condition",
    "sample",
    "InferResult",
    "Inferer",
]