"""
Non-linear optimization of RF models.

Provides solvers and routines to find the optimal point-estimates 
that minimize a given objective/cost function.
"""

# General API
from pmrf.optimize.base import (
    AbstractBoundedMinimizer,
    AbstractUnconstrainedMinimizer,
    AbstractMinimizer,
    is_optimizer,
    is_minimizer,
)
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.minimize import minimize

# Backends
from pmrf.optimize.solvers import optimistix, jaxopt, scipy
from pmrf.optimize.solvers.scipy import ScipyMinimize
from pmrf.optimize.solvers.optimistix import OptimistixMinimise
from pmrf.optimize.solvers.jaxopt import LBFGSB
from pmrf.optimize.solvers.optimistix import (
    NelderMead,
    LBFGS,
    BFGS,
    GradientDescent,
)

__all__ = [
    "AbstractBoundedMinimizer",
    "AbstractUnconstrainedMinimizer",
    "AbstractMinimizer",
    "is_optimizer",
    "is_minimizer",
    "OptimizeResult",
    "minimize",
    "scipy",
    "optimistix",
    "jaxopt",
    "ScipyMinimize",
    "OptimistixMinimise",
    "LBFGSB",
    "NelderMead",
    "LBFGS",
    "BFGS",
    "GradientDescent",
]