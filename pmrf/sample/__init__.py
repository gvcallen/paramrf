"""
The sampling module, for random and adaptive sampling of ParamRF models.

All samplers in this module inherit from :class:`pmrf.sample.BaseSampler`.
Sampling is done by initializing a Sampler class with the model and sampling targets, and then calling
:meth:`pmrf.sample.BaseSampler.run`. Sampling targets include the model features (e.g. S11),
the sampling frequency, and additional targets provided by specific sub-class algorithms
(such as surrogate models etc.).

When calling `run`, all key-word arguments are forwarded to the underlying backend/algorithm (for example, the EqxLearnUncertaintySampler).
This allows full configuration of the sampling algorithm, while also providing a convenience wrapper for simple use.

Results are returned in the form of :class:`pmrf.sample.SampleResults`. This contains the details about the initial configuration,
as well as the sampled parameters and features.
"""

from pmrf.sample.base import BaseSampler
from pmrf.sample.results import SampleResults
from pmrf.sample.oneshot import OneshotSampler
from pmrf.sample.acqusition import AcquisitionSampler
from pmrf.sample.algorithms import *
from pmrf.sample.backends import *

from pmrf.sample import algorithms, backends

__all__ = [
    "BaseSampler",
    "SampleResults",
    "OneshotSampler",
    "AcquisitionSampler",
]
__all__.extend(algorithms.__all__)
__all__.extend(backends.__all__)