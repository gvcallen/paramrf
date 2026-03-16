"""
Optimizer wrappers for interfacing with external libraries.
"""

__all__ = []

try:
    from pmrf.optimize.backends.scipy import SciPyMinimizeOptimizer
    __all__.append("SciPyMinimizeOptimizer")
except ImportError:
    pass

try:
    from pmrf.optimize.backends.optax import OptaxOptimizer
    __all__.append("OptaxOptimizer")
except ImportError:
    pass

try:
    from infer.backends.polychord import PolyChordOptimizer
    __all__.append("PolyChordOptimizer")
except ImportError:
    pass