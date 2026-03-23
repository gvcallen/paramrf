"""
Bayesian inference module for parametric RF models.

Provides samplers and routines to compute the posterior joint 
distributions of model parameters given measured data.
"""

from pmrf.infer.fit import fit, fit_sequential
from pmrf.infer.sample import sample
from pmrf.infer.result import InferResult
from pmrf.constants import Inferer

__all__ = [
    "fit",
    "fit_sequential",
    "sample",
    "InferResult",
    "Inferer",
]