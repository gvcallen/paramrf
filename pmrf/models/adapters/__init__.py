"""
Adapters for interfacing ParamRF models with external tools and formats.

This module provides the necessary wrappers to bridge the gap between 
ParamRF's internal representations and external standards, software, and libraries.
This includes scikit-rf Networks, EM simulation software, and generic Equinox models.
"""

from pmrf.models.adapters.base import Adapter, Discrete, SingleProperty, SingleDiscreteProperty
from pmrf.models.adapters.bridge import Host
from pmrf.models.adapters.collection import ListModel, DictModel
from pmrf.models.adapters.static import Measured
from pmrf.models.adapters.surrogate import ContinuousSurrogate, DiscreteSurrogate