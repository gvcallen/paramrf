"""
Unified routing module for model fitting.

Provides a high-level `fit` and `fit_sequential` interface that dynamically 
delegates to either frequentist optimization (`pmrf.optimize`) or Bayesian 
inference (`pmrf.infer`) based on the provided engine.
"""

from pmrf.fit.fit import fit, fit_sequential

__all__ = [
    "fit",
    "fit_sequential",
]