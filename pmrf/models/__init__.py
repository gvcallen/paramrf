"""
The core models library.

This module contains the core ParamRF :class:`Model` class, as well as various components, composite building models, adapters and numerical models.

**NB: Models are re-exported** at root :mod:`pmrf.models`.
"""

from pmrf.core import Model
from pmrf.core.adapters.base import Discrete, SingleProperty, SingleDiscreteProperty
from pmrf.core.adapters.bridge import Host
from pmrf.core.adapters.collection import ListModel, DictModel
from pmrf.core.adapters.static import Measured
from pmrf.core.adapters.surrogate import ContinuousSurrogate, DiscreteSurrogate
from pmrf.core.components.ideal import Port, Ground, SourceConverter, Transformer
from pmrf.core.components.lines import TransmissionLine, RLGCLine, PhaseLine, ConstantRLGCLine, PhysicalLine, DatasheetLine, CoaxialLine, MicrostripLine, ProfiledLine
from pmrf.core.components.lumped import Load, Resistor, Capacitor, Inductor, ShuntResistor, ShuntCapacitor, ShuntInductor, CapacitorQ, InductorQ, SHORT, OPEN, MATCH
from pmrf.core.components.nonideal import NonIdealResistor, CLCResistor
from pmrf.core.components.topological import PiCLC, BoxCLCC
from pmrf.core.composite.interconnected import Circuit, Cascade, Terminated, Shunt
from pmrf.core.composite.transformed import Renumbered, Flipped, Stacked
from pmrf.core.numerical.expansion import VectorExpansion

from pmrf.core import adapters, components, composite, numerical

__all__ = [
    "adapters",
    "components",
    "composite",
    "numerical",
]