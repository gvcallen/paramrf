"""
Optimization using SciPy or Optimistix.

Provides solvers and routines to find the optimal point-estimates 
that minimize a given objective/cost function.
"""

from pmrf.optimize.minimize import minimize
from pmrf.optimize.base import is_optimizer, is_minimizer, OptimizeResult
from pmrf.optimize.scipy import ScipyMinimize

from pmrf.constants import Optimizer

__all__ = [
    "minimize",
    "is_minimizer",
    "is_optimizer",
    "ScipyMinimize",
    "OptimizeResult",
    "Optimizer",
]