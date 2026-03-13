"""
The optimize module, for optimizing ParamRF models to meet some goal function.

All optimizers in this module inherit from :class:`pmrf.fitting.BaseOptimizer`.
Optimization is done by initializing an Optimizer class with the model and optimization goals, and then calling
:meth:`pmrf.optimize.BaseOptimizer.run`.

When calling `run`, all key-word arguments are forwarded to the underlying backend (for example, SciPy).
This allows full configuration of the optimization algorithm, while also providing a convenience wrapper
for simple use.

Results are returned in the form of :class:`pmrf.optimize.OptimizeResults`.
"""

from pmrf.optimize.base import BaseOptimizer
from pmrf.optimize.goal import Goal
from pmrf.optimize.results import OptimizeResults
from pmrf.optimize.frequentist import FrequentistOptimizer

from pmrf.optimize.backends import *
from pmrf.optimize import backends

__all__ = [
    "BaseFitter",
    "FitResults",
    "FrequentistFitter",
    "BayesianFitter",
]
__all__.extend(backends.__all__)