"""
Adapters for interfacing ParamRF models with external tools and formats.

This provides the necessary wrappers to bridge the gap between 
ParamRF's internal representations and external standards, software, and libraries.
This includes scikit-rf Networks, EM simulation software, and generic Equinox models.
"""

from pmrf.core.adapters import base, bridge, collection, static, surrogate

__all__ = [
    "base",
    "bridge",
    "collection",
    "static",
    "surrogate",
]