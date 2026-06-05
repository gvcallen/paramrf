"""
Adapters for interfacing ParamRF models with external tools and formats.

This provides the necessary wrappers to bridge the gap between 
ParamRF's internal representations and external standards, software, and libraries.
This includes scikit-rf Networks, EM simulation software, and generic Equinox models.
"""

from pmrf.models.adapters import base
from pmrf.models.adapters import bridge, static, callable

__all__ = [
    "base",
    "bridge",
    "static",
    "callable",
]

__sphinx_group__ = True