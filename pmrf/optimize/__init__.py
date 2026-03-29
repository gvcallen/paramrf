"""
Frequentist optimization module for parametric RF models.

Provides solvers and routines to find the optimal point-estimates 
that minimize a given objective/cost function.
"""

from pmrf.optimize.minimize import minimize, is_optimizer
from pmrf.optimize.fit import fit, fit_sequential
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers import ScipyMinimizer

from pmrf.constants import Optimizer

__all__ = [
    "minimize",
    "fit",
    "fit_sequential",
    "ScipyMinimizer",
    "OptimizeResult",
    "Optimizer",
]