"""
Unified routing module for model fitting.

Provides a high-level `fit` and `fit_sequential` interface that dynamically 
delegates to either frequentist optimization (`pmrf.optimize`) or Bayesian 
inference (`pmrf.infer`) based on the provided engine.

Note that both `fit` and `fit_sequential` are re-exported directly under `pmrf`.
"""

from pmrf.infer import is_inferer, InferResult
from pmrf.optimize import is_optimizer, OptimizeResult
from pmrf.fitting.fit import fit, fit_sequential
from pmrf.fitting.result import FitResult
from pmrf.constants import Optimizer, Inferer, Solver


__all__ = [
    "is_optimizer",
    "is_inferer",
    "InferResult",
    "OptimizeResult",
    "fit",
    "fit_sequential",
    "Optimizer",
    "Inferer",
    "Solver",
    "FitResult",
]