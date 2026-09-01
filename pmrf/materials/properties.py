r"""Frequency-evaluated, geometry-free material properties."""
import equinox as eqx

import jax.numpy as jnp
from scipy.constants import mu_0


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
    r"""Evaluated conductor properties.

    ``zs`` and :meth:`gamma` are independent: ``zs`` is the surface
    prefactor, which a surface treatment such as roughness may scale, while
    :meth:`gamma` describes diffusion into the unmodified bulk.

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

    def gamma(self, w) -> jnp.ndarray:
        r"""
        Return the propagation constant inside the metal, in 1/m.

        **Mathematical Formulation**

        $$\gamma = \sqrt{j\omega\mu\sigma} = \frac{1+j}{\delta},
        \qquad \delta = \sqrt{\frac{2}{\omega\mu\sigma}},$$

        with $\mu=\mu_0\mu_r$. This is the inverse complex skin depth: it
        governs how fast the field diffuses into the bulk, and it is what
        makes $\gamma a$ and $\gamma t$ the dimensionless "how many skin
        depths across is this cross-section" arguments of the Bessel and
        $\coth$ expressions of
        :mod:`~pmrf.materials.conductor_shape`. Despite the name it is not
        the propagation constant of the line.

        It is deliberately computed from $\sigma$ and $\mu_r$ rather than
        recovered from ``zs`` as $\sigma\zeta_c$. That identity holds only
        for a smooth bulk metal: any surface treatment which scales ``zs``
        -- roughness today, cladding or plating tomorrow -- would otherwise
        inflate the diffusion constant, which surface texture does not
        change.

        The value is zero at dc, where the $\sqrt{\cdot}$ branch point is
        guarded so the gradient stays finite, and infinite for a perfect
        conductor. Callers are responsible for their own dc and
        perfect-conductor branches.

        Parameters
        ----------
        w : ArrayLike
            Angular frequency in rad/s.

        Returns
        -------
        jnp.ndarray
            Propagation constant inside the metal, in 1/m.

        References
        ----------
        Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 1.7.
        Wiley.
        """
        w = jnp.asarray(w)
        safe_w = jnp.where(w > 0, w, 1.0)
        k = jnp.sqrt(safe_w * mu_0 * self.mu_r * self.sigma / 2)
        return jnp.where(w > 0, k * (1 + 1j), 0.0)
