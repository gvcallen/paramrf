"""
Optimizer wrappers for interfacing with external libraries.
"""

__all__ = []

try:
    from pmrf.optimize.backends.scipy import SciPyMinimizeOptimizer
    __all__.append("SciPyMinimizeOptimizer")
except ImportError:
    pass