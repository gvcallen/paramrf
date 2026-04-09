"""
Composite models that wrap and manipulate other models to create new ones.

This includes cascading, port renumbering, complex circuit modeling, and more.
"""

from pmrf.models.composite import arranged as arranged
from pmrf.models.composite import collection as collection
from pmrf.models.composite import interconnected as interconnected
from pmrf.models.composite import transformed as transformed

__all__ = [
    "arranged",
    "collection",
    "interconnected",
    "transformed",
]