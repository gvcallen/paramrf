"""
Composite models that wrap and manipulate other models to create new ones.

This includes cascading, port renumbering, complex circuit modeling, and more.
"""

from pmrf.models.composite import interconnected as interconnected
from pmrf.models.composite import transformed as transformed
from pmrf.models.composite import nodal as nodal
from pmrf.models.composite import wrapped as wrapped

__all__ = [
    "interconnected",
    "transformed",
    "nodal",
    "wrapped",
]