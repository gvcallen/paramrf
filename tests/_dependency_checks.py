"""Capability-based skips for optional or version-sensitive test dependencies."""

import pytest


def _has_attribute(module_name: str, attribute: str) -> bool:
    try:
        module = __import__(module_name, fromlist=[attribute])
    except ImportError:
        return False
    return hasattr(module, attribute)


requires_distreqx_transpose = pytest.mark.skipif(
    not _has_attribute("distreqx.bijectors", "Transpose"),
    reason="requires distreqx.bijectors.Transpose",
)

requires_distreqx_joint = pytest.mark.skipif(
    not _has_attribute("distreqx.distributions", "Joint"),
    reason="requires distreqx.distributions.Joint",
)
