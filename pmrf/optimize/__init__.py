"""
Optimization using SciPy or Optimistix.

Provides solvers and routines to find the optimal point-estimates 
that minimize a given objective/cost function.
"""

from pmrf.optimize.minimize import minimize
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers import ScipyMinimizer, is_optimizer

from pmrf.constants import Optimizer

__all__ = [
    "minimize",
    "is_optimizer",
    "ScipyMinimizer",
    "OptimizeResult",
    "Optimizer",
]