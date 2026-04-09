"""
Higher-level fitting utilities.

Provides high-level fitting interfaces that automatically delgate to specific solvers,
currently either frequentist optimization (`pmrf.optimize`) or Bayesian 
inference (`pmrf.infer`).

Note that both `fit` and `fit_sequential` are re-exported directly under `pmrf`.
"""

from pmrf.fitting.fit import fit, fit_sequential
from pmrf.fitting.minimize import fit_minimize
from pmrf.fitting.sample import fit_sample
from pmrf.fitting.result import FitResult


__all__ = [
    "fit",
    "fit_sequential",
    "fit_minimize",
    "fit_sample",
    "FitResult",
]