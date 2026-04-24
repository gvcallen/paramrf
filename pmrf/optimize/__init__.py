"""
Optimization using Optimistix or SciPy.

Provides solvers and routines to find the optimal point-estimates 
that minimize a given objective/cost function.
"""

from pmrf.optimize.minimize import minimize
from pmrf.optimize.base import is_optimizer, is_minimizer, OptimizeResult, AbstractCallableMinimizer
from pmrf.optimize.scipy import ScipyMinimize

__all__ = [
    "minimize",
    "is_minimizer",
    "is_optimizer",
    "ScipyMinimize",
    "OptimizeResult",
    "Optimizer",
    "AbstractCallableMinimizer",
]