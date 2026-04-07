"""
Higher-level fitting utilities.

Provides a high-level `fit` and `fit_sequential` interface that dynamically 
delegates to either frequentist optimization (`pmrf.optimize`) or Bayesian 
inference (`pmrf.infer`) based on the provided engine.

Note that both `fit` and `fit_sequential` are re-exported directly under `pmrf`.
"""

from pmrf.infer import is_inferer as is_inferer, InferResult as InferResult
from pmrf.optimize import is_optimizer as is_optimizer, OptimizeResult as OptimizeResult
from pmrf.fitting.fit import fit, fit_sequential
from pmrf.fitting.result import FitResult
from pmrf.constants import Optimizer, Inferer, Solver


__all__ = [
    "fit",
    "fit_sequential",
    "Optimizer",
    "Inferer",
    "Solver",
    "FitResult",
]