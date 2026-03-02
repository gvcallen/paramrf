"""
The **fitting** module, for fitting ParamRF models to measured data.

All fitters in this module inherit from :class:`~pmrf.sampling.BaseFitter`.
Fitting is done by initializing a Fitter class with the model and fitting goals, and then calling
:meth:`run <pmrf.fitting.BaseFitter.run>` with measured data. Fitting goals include the features to
fit to (e.g. S11), the fit frequency, the cost function (for frequentist fitters) or
the likelihood function (for Bayesian fitters).

When calling `run`, all key-word arguments are forwarded to the underlying backend (for example, PolyChord or SciPy).
This allows full configuration of the optimization algorithm, while also providing a convenience wrapper
for simple use.

Results are returned in the form of :class:`~pmrf.fitting.FitResults`. This contains the details about the initial configuration,
as well as the sampled parameters and features.
"""

from pmrf.fitting.base import BaseFitter
from pmrf.fitting.results import FitResults
from pmrf.fitting.frequentist import FrequentistFitter
from pmrf.fitting.bayesian import BayesianFitter

from pmrf.fitting.backends import *
from pmrf.fitting import backends

__all__ = [
    "BaseFitter",
    "FitResults",
    "FrequentistFitter",
    "BayesianFitter",
]
__all__.extend(backends.__all__)