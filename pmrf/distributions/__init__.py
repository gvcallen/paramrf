"""
Distribution wrappers for interfacing with external libraries.
"""

__all__ = []

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