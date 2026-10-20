"""Latent Gaussian processes and their event-space mixing maps."""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp

from pmrf.modules.base import Module
from pmrf.utils import field


class SharedLatent(Module):
    """A shared latent with independent kernel rows and a dataset mixing.

    The kernel's batch shape determines the number of rows. ``mixing`` accepts
    ``(prediction_event, x)`` and returns a real ``(N, blocks, rows)`` array.

    Parameters
    ----------
    kernel : Callable
        Covariance kernel; its batch shape determines the latent row count.
    mixing : Callable
        Map from a dataset's event-space prediction and frequencies to its
        ``(N, blocks, rows)`` real mixing matrix.
    jitter : float, default=1e-10
        Diagonal stabilization for each latent row's covariance.
    """

    #: Covariance kernel for each independent latent row.
    kernel: Callable

    #: Map from latent rows to a dataset's flattened event blocks.
    mixing: Callable

    #: Diagonal stabilization of the latent covariance.
    jitter: float = field(default=1e-10, static=True)


class IdentityMixing(eqx.Module):
    """Map each latent row to the corresponding flattened event block.

    The number of rows must equal the number of event blocks.
    """

    def __call__(self, prediction_event: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return an identity matrix at each frequency.

        Parameters
        ----------
        prediction_event : jnp.ndarray
            Event-space prediction, with frequency last.
        x : jnp.ndarray
            Frequency points.

        Returns
        -------
        jnp.ndarray
            Identity mixing of shape ``(N, blocks, blocks)``.
        """
        blocks = prediction_event.size // prediction_event.shape[-1]
        return jnp.broadcast_to(jnp.eye(blocks), (len(x), blocks, blocks))


class ComplexPowerMixing(eqx.Module):
    r"""Multiply a complex latent by the event prediction raised to ``power``.

    Assumes the default Re/Im event layout ``(..., 2, N)``. Each complex
    prediction block reads the same two latent rows. ``power=0`` gives a
    complex additive latent.

    Parameters
    ----------
    power : float, default=1.0
        Exponent applied to each complex event-space prediction.
    """

    #: Exponent of the complex event-space prediction.
    power: float = field(default=1.0, static=True)

    def __call__(self, prediction_event: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return the real two-row map for complex multiplication.

        Parameters
        ----------
        prediction_event : jnp.ndarray
            Prediction with default Re/Im event layout ``(..., 2, N)``.
        x : jnp.ndarray
            Frequency points.

        Returns
        -------
        jnp.ndarray
            Mixing of shape ``(N, blocks, 2)``.
        """
        if prediction_event.ndim < 2 or prediction_event.shape[-2] != 2:
            raise ValueError("ComplexPowerMixing requires default Re/Im event layout (..., 2, N).")
        pairs = prediction_event.reshape((-1, 2, len(x)))
        complex_prediction = pairs[:, 0, :] + 1j * pairs[:, 1, :]
        factor = complex_prediction ** self.power
        real, imag = jnp.real(factor).T, jnp.imag(factor).T
        matrix = jnp.stack((
            jnp.stack((real, -imag), axis=-1),
            jnp.stack((imag, real), axis=-1),
        ), axis=-2)
        return matrix.reshape((len(x), -1, 2))


__all__ = ['SharedLatent', 'IdentityMixing', 'ComplexPowerMixing']
