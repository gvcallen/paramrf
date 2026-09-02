"""Typed cross-section records for planar transmission lines.

A cross-section record is the geometry a line hands to a current-distribution
strategy: one frozen record per planar family, because no stable set of
cross-section quantities exists *across* families -- coplanar waveguide brings
a gap width, suspended substrate an air gap, offset stripline an offset --
while each family's own set is stable.

The record carries dimensions only. Quantities that are solved rather than
given -- the characteristic impedance, the effective permittivity -- reach a
strategy through the quasi-static result instead, so a strategy that consumes
solved intermediates needs no new record field.
"""
import equinox as eqx
import jax.numpy as jnp


class AbstractPlanarCrossSection(eqx.Module):
    """Frozen cross-section geometry of one planar line family.

    Subclasses are per-family: pairing a strategy written for one family with
    another family's record is a type error at the current-distribution
    boundary rather than a missing-argument error deep in a call.
    """

    def dimensions(self) -> dict:
        """Return the dimensions a conductor shape may be solved with.

        Conductor shapes are shared across families and name their own
        geometry, so the record translates itself into that vocabulary once,
        here, rather than at each call site.
        """
        raise NotImplementedError


class MicrostripCrossSection(AbstractPlanarCrossSection):
    """Cross-section of a strip on a grounded dielectric sheet.

    Parameters
    ----------
    w : ArrayLike
        Width of the strip in meters.
    h : ArrayLike
        Substrate height in meters. Edge-shape and ground-plane-share loss
        models need it even though the classical incremental-inductance rule
        does not.
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
        Complex relative permittivity of the homogeneous filling. It is a
        material property rather than a dimension, but Cohn's attenuation
        branches select on it, so the stripline record carries it.
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
