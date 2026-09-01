"""Frequency-evaluated, geometry-free material properties."""
import equinox as eqx

import jax.numpy as jnp


class DielectricProperties(eqx.Module):
    """Evaluated dielectric properties.

    Parameters
    ----------
    ep_r : jnp.ndarray
        Complex relative permittivity excluding static conduction.
    mu_r : jnp.ndarray
        Complex relative permeability.
    sigma : jnp.ndarray
        Static bulk conductivity in S/m.
    """

    ep_r: jnp.ndarray
    mu_r: jnp.ndarray
    sigma: jnp.ndarray


class ConductorProperties(eqx.Module):
    """Evaluated conductor properties.

    Parameters
    ----------
    zs : jnp.ndarray
        Surface impedance in ohm per square.
    sigma : jnp.ndarray
        Bulk conductivity in S/m.
    mu_r : jnp.ndarray
        Complex relative permeability.
    """

    zs: jnp.ndarray
    sigma: jnp.ndarray
    mu_r: jnp.ndarray
