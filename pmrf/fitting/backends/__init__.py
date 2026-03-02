"""
Fitter wrappers for interfacing with external libraries.
"""

__all__ = []

try:
    from pmrf.fitting.backends.blackjax import BlackJAXNSFitter
    __all__.append("BlackJAXNSFitter")
except ImportError:
    pass

try:
    from pmrf.fitting.backends.numpyro import NumPyroFitter, NumPyroNSFitter, NumPyroMCMCFitter
    __all__.extend([
        "NumPyroFitter", 
        "NumPyroNSFitter", 
        "NumPyroMCMCFitter"
    ])
except ImportError:
    pass

try:
    from pmrf.fitting.backends.polychord import PolyChordFitter
    __all__.append("PolyChordFitter")
except ImportError:
    pass

try:
    from pmrf.fitting.backends.scipy import SciPyMinimizeFitter
    __all__.append("SciPyMinimizeFitter")
except ImportError:
    pass