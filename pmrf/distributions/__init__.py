"""
Distribution wrappers for interfacing with external libraries.
"""

__all__ = []

from pmrf.distributions.empirical import Empirical
__all__.append("Empirical")

try:
    from pmrf.distributions.anesthetic import AnestheticDistribution
    __all__.append("AnestheticDistribution")
except ImportError:
    pass

try:
    from pmrf.distributions.flowjax import FlowJAXDistribution
    __all__.append("FlowJAXDistribution")
except ImportError:
    pass
