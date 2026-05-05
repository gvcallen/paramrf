"""
Optimization using Optimistix or SciPy.

Provides solvers and routines to find the optimal point-estimates 
that minimize a given objective/cost function.
"""

# Base functions
from pmrf.optimize.base import (
    AbstractBoundedMinimizer as AbstractBoundedMinimizer,
    AbstractUnconstrainedMinimizer as AbstractUnconstrainedMinimizer,
    AbstractMinimizer as AbstractMinimizer,
    is_optimizer as is_optimizer,
    is_minimizer as is_minimizer,
    OptimizeResult as OptimizeResult,
)

# Main minimize
from pmrf.optimize.minimize import minimize

# Backend wrappers
from pmrf.optimize.scipy import ScipyMinimize
from pmrf.optimize.optimistix import OptimistixMinimise

# Specific algorithm re-exports
from pmrf.optimize.jaxopt import (
    LBFGSB,
)

# General module re-exports
from pmrf.optimize import (
    base,
    scipy,
    optimistix,
    jaxopt,
)

__all__ = [
    "AbstractBoundedMinimizer",
    "AbstractUnconstrainedMinimizer",
    "AbstractMinimizer",
    "is_optimizer",
    "is_minimizer",
    "OptimizeResult",
    "minimize",
    "ScipyMinimize",
    "OptimistixMinimise",
    "LBFGSB",
    "base",
    "scipy",
    "optimistix",
    "jaxopt",
]