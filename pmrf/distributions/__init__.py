"""
Distribution wrappers for interfacing with external libraries.
"""

from pmrf.distributions.empirical import Empirical
from pmrf.distributions.flowjax import FlowJAXDistribution

__all__ = [
    "Empirical",
    "FlowJAXDistribution"
]
