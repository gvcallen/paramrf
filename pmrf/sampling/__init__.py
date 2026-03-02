"""
The sampling module, for random and adaptive sampling of ParamRF models.

All samplers in this module inherit from :class:`pmrf.sampling.BaseSampler`.
Sampling is done by initializing a Sampler class with the model and sampling targets, and then calling
:meth:`run pmrf.sampling.BaseSampler.run`. Sampling targets include the model features (e.g. S11),
the sampling frequency, and additional targets provided by specific sub-class algorithms
(such as surrogate models etc.).

When calling `run`, all key-word arguments are forwarded to the underlying backend/algorithm (for example, the EqxLearnUncertaintySampler).
This allows full configuration of the sampling algorithm, while also providing a convenience wrapper for simple use.

Results are returned in the form of :class:`pmrf.sampling.SampleResults`. This contains the details about the initial configuration,
as well as the sampled parameters and features.
"""

from pmrf.sampling.base import BaseSampler
from pmrf.sampling.results import SampleResults
from pmrf.sampling.oneshot import OneshotSampler
from pmrf.sampling.acqusition import AcquisitionSampler
from pmrf.sampling.algorithms import *
from pmrf.sampling.backends import *

from pmrf.sampling import algorithms, backends

__all__ = [
    "BaseSampler",
    "SampleResults",
    "OneshotSampler",
    "AcquisitionSampler",
]
__all__.extend(algorithms.__all__)
__all__.extend(backends.__all__)