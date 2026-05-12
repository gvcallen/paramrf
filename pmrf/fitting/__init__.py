"""
Higher-level fitting utilities.

Provides high-level fitting interfaces that automatically delgate to specific solvers,
currently either frequentist optimization (`pmrf.optimize`) or Bayesian 
inference (`pmrf.infer`).
"""

from pmrf.fitting.result import FitResult
from pmrf.fitting.minimize import fit_minimize
from pmrf.fitting.sample import fit_sample
from pmrf.fitting.routers import fit, fit_sequential, fit_joint, AbstractFitter

__all__ = [
    "fit",
    "fit_sequential",
    "fit_joint",
    "fit_minimize",
    "fit_sample",
    "AbstractFitter",
    "FitResult",
    "Solver",
]