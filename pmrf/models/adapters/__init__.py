"""
Adapters for interfacing ParamRF models with external tools and formats.

This provides the necessary wrappers to bridge the gap between 
ParamRF's internal representations and external standards, software, and libraries.
This includes scikit-rf Networks, EM simulation software, and generic Equinox models.
"""

from pmrf.models.adapters import base
from pmrf.models.adapters import bridge, static, callable
from pmrf.models.composite import collection

__all__ = [
    "base",
    "bridge",
    "collection",
    "static",
    "callable",
]