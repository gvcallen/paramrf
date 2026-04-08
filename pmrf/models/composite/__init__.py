"""
Composite "building blocks" that contain other models.

This module allows for combining and transformed existing models to create new ones.
For example, cascading, renumbering ports, or creating complex circuit models.
"""

from pmrf.models.composite import interconnected, transformed

__all__ = [
    "collection",
    "interconnected",
    "transformed",
]