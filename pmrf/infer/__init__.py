"""
Optimizer wrappers for interfacing with external libraries.
"""

__all__ = []

try:
    from pmrf.infer.backends.polychord import sample_polychord
    __all__.append("sample_polychord")
except ImportError:
    pass