"""
This module provides built-in numerical models in ParamRF.
"""

from pmrf.models.surrogates import expansion as expansion
from pmrf.models.surrogates import rational as rational

__all__ = [
    "expansion",
    "rational",
]