"""Cross-section records for planar transmission lines.

Each record contains the fixed geometry required by one planar line family.
Solved quantities are supplied separately in the quasi-static result.
"""
import equinox as eqx
import jax.numpy as jnp


class AbstractPlanarCrossSection(eqx.Module):
    """Cross-section geometry for a planar line family."""

    def dimensions(self) -> dict:
        """Return dimensions accepted by conductor-shape formulations."""
        raise NotImplementedError


class MicrostripCrossSection(AbstractPlanarCrossSection):
    """Cross-section of a strip on a grounded dielectric sheet.

    Parameters
    ----------
    w : ArrayLike
        Width of the strip in meters.
    h : ArrayLike
        Substrate height in meters.
    t : ArrayLike | None, default=None
        Strip thickness in meters, or ``None`` when it is unspecified.
    """

    #: Width of the strip in meters
    w: jnp.ndarray

    #: Substrate height in meters
    h: jnp.ndarray

    #: Strip thickness in meters, or ``None`` when unspecified
    t: jnp.ndarray | None = None

    def dimensions(self) -> dict:
        return {"w": self.w, "t": self.t}


class StriplineCrossSection(AbstractPlanarCrossSection):
    """Cross-section of a centre strip between two ground planes.

    Parameters
    ----------
    w : ArrayLike
        Width of the centre strip in meters.
    b : ArrayLike
        Separation of the ground planes in meters.
    t : ArrayLike | None, default=None
        Strip thickness in meters, or ``None`` when it is unspecified.
    ep_r : jnp.ndarray | None, default=None
        Complex relative permittivity of the homogeneous filling, used by
        Cohn's attenuation model.
    """

    #: Width of the centre strip in meters
    w: jnp.ndarray

    #: Separation of the ground planes in meters
    b: jnp.ndarray

    #: Strip thickness in meters, or ``None`` when unspecified
    t: jnp.ndarray | None = None

    #: Complex relative permittivity of the filling
    ep_r: jnp.ndarray | None = None

    def dimensions(self) -> dict:
        return {"w": self.w, "t": self.t}
