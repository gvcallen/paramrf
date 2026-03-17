"""
Optimizer wrappers for interfacing with external libraries.
"""

__all__ = []

try:
    from pmrf.optimize.backends.optimistix import OptimistixOptimizer
    __all__.append("OptimistixOptimizer")
except ImportError:
    pass

try:
    from pmrf.optimize.backends.scipy import ScipyOptimizer
    __all__.append("ScipyOptimizer")
except ImportError:
    pass