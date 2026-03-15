"""
Generic backend wrappers for interfacing with external libraries.
"""

__all__ = []

try:
    from pmrf.backends.scipy import run_scipy_minimize
    __all__.append("run_scipy_minimize")
except ImportError:
    pass

try:
    from pmrf.backends.optax import run_optax
    __all__.append("run_optax")
except ImportError:
    pass
