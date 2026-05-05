"""
Optimization using Optimistix or SciPy.

Provides solvers and routines to find the optimal point-estimates 
that minimize a given objective/cost function.
"""

# Base functions
from pmrf.optimize.backends import jaxopt, optimistix, scipy
from pmrf.optimize.base import (
    AbstractBoundedMinimizer as AbstractBoundedMinimizer,
    AbstractUnconstrainedMinimizer as AbstractUnconstrainedMinimizer,
    AbstractMinimizer as AbstractMinimizer,
    is_optimizer as is_optimizer,
    is_minimizer as is_minimizer,
)
from pmrf.optimize.result import OptimizeResult as OptimizeResult

# Main minimize
from pmrf.optimize.minimize import minimize

# Backends
from pmrf.optimize.backends import (
    scipy,
    optimistix,
    jaxopt
)

# Specific backend wrappers
from pmrf.optimize.backends.scipy import ScipyMinimize
from pmrf.optimize.backends.optimistix import OptimistixMinimise

# Specific algorithm re-exports
from pmrf.optimize.backends.jaxopt import (
    LBFGSB,
)
from pmrf.optimize.backends.optimistix import (
    NelderMead,
    LBFGS,
    BFGS,
    GradientDescent,
)

# General module re-exports
from pmrf.optimize import (
    base,
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
    "LBFGS",
    "BFGS",
    "GradientDescent",
    "NelderMead",
    "base",
    "scipy",
    "optimistix",
    "jaxopt",
]