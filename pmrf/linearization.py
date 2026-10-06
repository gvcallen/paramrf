r"""
The linearisation of a fit at its MAP, and the linearised posterior covariance.

A linearisation is a Gauss–Newton approximation of a fit: the Jacobian
$J = -\partial r / \partial \theta$ of the event-space residual with respect to the
model's free parameters, with the discrepancy's hyperparameters held fixed, and the
Fisher matrix $F = \sum_b J_b^\top \Sigma_{D,b}^{-1} J_b$ with
$\Sigma_D = K + \Sigma_n$ per event block. Build one with
:meth:`pmrf.evaluators.MarginalLogLikelihood.linearize`, and combine one or more with
the prior in :func:`posterior_covariance`.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from pmrf.discrepancy_models import _group_by_matrix
from pmrf.parameters import Space, _check_space, log_prior, update, values as _values
from pmrf.utils import field


class Linearization(eqx.Module):
    r"""The linearisation of one fit at a model.

    Built by :meth:`pmrf.evaluators.MarginalLogLikelihood.linearize`. Columns of
    :attr:`J`, and rows and columns of :attr:`F`, follow the free parameters in
    :attr:`names` order, each flattened in C order to its size in :attr:`shapes`.
    """

    #: The Jacobian $J = -\partial r / \partial \theta$ of the residual, with shape
    #: ``(*batch, N, p)``.
    J: Array

    #: The residual $r$ in event space, with shape ``(*batch, N)``.
    residual: Array

    #: The lower Cholesky factor of $\Sigma_D = K + \Sigma_n$, at its own batch shape,
    #: which broadcasts to ``batch``.
    chol: Array

    #: The Fisher matrix $F = \sum_b J_b^\top \Sigma_{D,b}^{-1} J_b$, with shape ``(p, p)``.
    F: Array

    #: The names of the free parameters, in column order.
    names: tuple[str, ...] = field(static=True)

    #: The shape of each free parameter's value, in :attr:`names` order.
    shapes: tuple[tuple[int, ...], ...] = field(static=True)

    #: The space the parameter values, and so :attr:`J` and :attr:`F`, are in.
    space: str = field(static=True)

    def unflatten(self, vector: Array) -> dict[str, Array]:
        """Map a flat vector over the parameters, such as a row of :attr:`F`, back to names."""
        return _unflatten(self.names, self.shapes, vector)


def _unflatten(names, shapes, vector) -> dict[str, Array]:
    sizes = [math.prod(shape) for shape in shapes]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    return {
        name: jnp.reshape(vector[..., offset:offset + size], vector.shape[:-1] + shape)
        for name, shape, offset, size in zip(names, shapes, offsets, sizes)
    }


def _flat(model: PyTree, space: Space):
    """The free parameter values of `model` in `space` as one vector, and its map back.

    Returns the names, the shapes, the flat vector, and a function from a flat vector to
    a copy of `model` holding those values.
    """
    _check_space(space)
    named = _values(model, free_only=True, space=space)
    if not named:
        raise ValueError("The model has no free parameters to linearise over.")
    names = tuple(named)
    shapes = tuple(tuple(jnp.shape(value)) for value in named.values())
    vector = jnp.concatenate([jnp.ravel(value) for value in named.values()])

    def rebuild(x):
        return update(model, _unflatten(names, shapes, x), space=space)

    return names, shapes, vector, rebuild


def _linearize(
    residual_fn: Callable[[PyTree], Array],
    model: PyTree,
    chol: Array,
    space: Space = 'declared',
) -> Linearization:
    """Linearise a residual function of a model, given the factor of its covariance.

    The evaluator-facing entry point is
    :meth:`pmrf.evaluators.MarginalLogLikelihood.linearize`, which supplies
    `residual_fn` and `chol`.

    Parameters
    ----------
    residual_fn : Callable
        Maps a model to its event-space residual, with shape ``(*batch, N)``.
    model : PyTree
        The model to linearise at. Must still be wrapped.
    chol : jax.Array
        The lower Cholesky factor of $\\Sigma_D$, with shape ``(*chol_batch, N, N)``
        broadcasting to ``batch``.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space of the parameter values differentiated with respect to.

    Returns
    -------
    Linearization
    """
    names, shapes, x0, rebuild = _flat(model, space)
    residual = residual_fn(model)
    J = -jax.jacfwd(lambda x: residual_fn(rebuild(x)))(x0)

    # Event blocks sharing a factor are solved together, with the parameter axis
    # carried as one more batch axis the factor is shared along.
    n, p = J.shape[-2], J.shape[-1]
    matrix_shape, R, _ = _group_by_matrix(chol.shape[:-2], jnp.moveaxis(J, -1, 0))
    L = jnp.broadcast_to(chol, chol.shape[:-2] + (n, n)).reshape(matrix_shape + (n, n))
    W = jsp.linalg.solve_triangular(L, R, lower=True)
    W = W.reshape(W.shape[:-1] + (p, -1))
    F = jnp.einsum('...nps,...nqs->pq', W, W)
    return Linearization(J, residual, chol, F, names, shapes, space)


def posterior_covariance(
    linearizations: Sequence[Linearization],
    model: PyTree,
    space: Space = 'declared',
) -> Array:
    r"""The linearised posterior covariance of a model's free parameters.

    **Mathematical Formulation**

    The Fisher matrices of independent fits add, and the prior enters through its
    precision at `model`:

    $$\Sigma_\text{post} = \left(\sum_k F_k + \Sigma_0^{-1}\right)^{-1}, \qquad
    \Sigma_0^{-1} = -\nabla^2 \log p(\theta)$$

    with the Hessian of :func:`pmrf.log_prior` taken in `space`. A parameter without a
    prior, or with a uniform prior inside its bounds, adds zero to $\Sigma_0^{-1}$.

    Parameters
    ----------
    linearizations : Sequence[Linearization]
        One linearisation per independent fit, each of `model` in `space`.
    model : PyTree
        The model the linearisations were taken at, carrying the prior. Must still be
        wrapped.
    space : {'declared', 'physical', 'raw'}, default='declared'
        The space of the parameter values. See :class:`pmrf.Param`.

    Returns
    -------
    jax.Array
        The covariance, with shape ``(p, p)``, over the free parameters in the
        linearisations' :attr:`~Linearization.names` order.

    Raises
    ------
    ValueError
        If no linearisation is given, or one is over other parameters or another space.
    """
    if not linearizations:
        raise ValueError("At least one linearisation is required.")
    names, shapes, x0, rebuild = _flat(model, space)
    for linearization in linearizations:
        if linearization.space != space:
            raise ValueError(
                f"A linearisation is in {linearization.space!r} space, not {space!r}."
            )
        if linearization.names != names or linearization.shapes != shapes:
            raise ValueError(
                "A linearisation is over different free parameters from the model: "
                f"{linearization.names} rather than {names}."
            )
    fisher = sum(linearization.F for linearization in linearizations)
    precision = -jax.hessian(lambda x: log_prior(rebuild(x), space=space))(x0)
    precision = fisher + precision
    precision = 0.5 * (precision + precision.T)
    return jnp.linalg.solve(precision, jnp.eye(precision.shape[-1], dtype=precision.dtype))


__all__ = [
    'Linearization',
    'posterior_covariance',
]
